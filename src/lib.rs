use indexmap::IndexMap;
use polars::datatypes::DataType as PolarsDataType;
use polars::export::arrow::array::{Array, Int32Array, Int64Array, StructArray};
use polars::export::arrow::chunk::Chunk;
use polars::export::arrow::datatypes::{DataType as ArrowDataType, Metadata};
use polars::export::arrow::io::ipc::read;
use polars::export::arrow::io::ipc::read::StreamMetadata;
use polars::export::chrono::NaiveDateTime;
use polars::prelude::{ArrayRef, DataFrame, NamedFrom, Result as PolarsResult, Series, TimeUnit};
use std::collections::HashMap;
use std::io::Read;

// do you like long names?
pub fn snowflake_arrow_ipc_streaming_binary_to_dataframe<R: Read>(
    arrow_data: &mut R,
) -> PolarsResult<DataFrame> {
    let (chunks, field_metadata, stream_metadata) = get_chunks_for_data(arrow_data);

    // Use indexmap for now, @todo convert to vec once the types are all in, such as binary
    let mut series_vec: IndexMap<usize, Series> = IndexMap::with_capacity(chunks.len());

    // create here, so we can use it in the loop
    for (field_index, field) in stream_metadata.schema.fields.iter().enumerate() {
        let fm = field_metadata.get(&field_index).unwrap();

        // @todo turn this into a macro
        match field.data_type() {
            ArrowDataType::Utf8 => {
                series_vec.insert(
                    field_index,
                    Series::new_empty(field.name.as_ref(), &PolarsDataType::Utf8),
                );
            }
            // DataType::Binary => {
            //     series_vec.insert(
            //         field_index,
            //         Series::new_empty(field.name.as_ref(), &PolarsDataType::Unknown),
            //     );
            // }
            ArrowDataType::Boolean => {
                series_vec.insert(
                    field_index,
                    Series::new_empty(field.name.as_ref(), &PolarsDataType::Boolean),
                );
            }
            ArrowDataType::Date32 => {
                series_vec.insert(
                    field_index,
                    Series::new_empty(field.name.as_ref(), &PolarsDataType::Date),
                );
            }
            ArrowDataType::Float64 => {
                series_vec.insert(
                    field_index,
                    Series::new_empty(field.name.as_ref(), &PolarsDataType::Float64),
                );
            }
            ArrowDataType::Int64 => {
                series_vec.insert(
                    field_index,
                    Series::new_empty(field.name.as_ref(), &PolarsDataType::Int64),
                );
            }
            ArrowDataType::Int32 => {
                if fm.get("scale").unwrap() == "0" {
                    series_vec.insert(
                        field_index,
                        Series::new_empty(field.name.as_ref(), &PolarsDataType::Int32),
                    );
                } else {
                    series_vec.insert(
                        field_index,
                        Series::new_empty(field.name.as_ref(), &PolarsDataType::Float64),
                    );
                }
            }
            ArrowDataType::Struct(_f) => {
                let logical_type = fm.get("logicalType").unwrap().as_str();
                match logical_type {
                    "TIMESTAMP_NTZ" | "TIMESTAMP_LTZ" | "TIMESTAMP_TZ" => {
                        // @todo split these into separate series
                        series_vec.insert(
                            field_index,
                            Series::new_empty(
                                field.name.as_ref(),
                                &PolarsDataType::Datetime(TimeUnit::Milliseconds, None),
                            ),
                        );
                    }
                    _ => unreachable!(),
                }
            }
            _dt => {
                // println!("{:?}", dt);
            }
        }
    }

    for chunk in chunks {
        for (array_index, array) in chunk.arrays().iter().enumerate() {
            let fm = field_metadata.get(&array_index).unwrap();
            // This might be dumb, but it works for now
            let field = stream_metadata.schema.fields.get(array_index).unwrap();

            // Once we figure out how to put binary values in polars (unknown datatype?, we can move found_series/clone_array up here.
            match array.data_type() {
                ArrowDataType::Utf8
                | ArrowDataType::Boolean
                | ArrowDataType::Float64
                | ArrowDataType::Date32
                | ArrowDataType::Int64 => {
                    // get it from the field_index_map
                    let found_series = series_vec.get_mut(&array_index).unwrap();

                    // is there a better way to do this without clone?
                    let cloned_array: ArrayRef = array.to_boxed();
                    let series = Series::try_from((field.name.as_ref(), cloned_array)).unwrap();
                    found_series.append(&series).expect("TODO: panic message");
                }
                ArrowDataType::Int32 => {
                    if fm.get("scale").unwrap() == "0" {
                        // get it from the field_index_map
                        let found_series = series_vec.get_mut(&array_index).unwrap();

                        // is there a better way to do this without clone?
                        let cloned_array: ArrayRef = array.to_boxed();
                        let series = Series::try_from((field.name.as_ref(), cloned_array)).unwrap();
                        found_series.append(&series).expect("TODO: panic message");
                    } else {
                        // build f64 from int32
                        let scale = fm.get("scale").unwrap().parse::<i32>().unwrap();
                        let found_series = series_vec.get_mut(&array_index).unwrap();

                        let f64 = array
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .unwrap()
                            .iter()
                            .map(|v| v.map(|x| *x as f64 / 10f64.powi(scale) as f64))
                            .collect::<Vec<Option<f64>>>();
                        // convert it to series?
                        let series = Series::new(field.name.as_ref(), &f64);
                        found_series.append(&series).expect("TODO: panic message");
                    }
                }
                ArrowDataType::Struct(_f) => {
                    let logical_type = fm.get("logicalType").unwrap().as_str();
                    let _cloned_array: ArrayRef = array.to_boxed();
                    let found_series = series_vec.get_mut(&array_index).unwrap();

                    match logical_type {
                        "TIMESTAMP_NTZ" | "TIMESTAMP_LTZ" | "TIMESTAMP_TZ" => {
                            let (_fields, struct_arrays, _bitmap) = array
                                .as_any()
                                .downcast_ref::<StructArray>()
                                .unwrap()
                                .clone()
                                .into_data();

                            // Get the fractional parts from the second array
                            let fractions = struct_arrays[1]
                                .as_any()
                                .downcast_ref::<Int32Array>()
                                .unwrap();

                            let datetimes: Vec<Option<NaiveDateTime>> = struct_arrays[0]
                                .as_any()
                                .downcast_ref::<Int64Array>()
                                .unwrap()
                                .iter()
                                .enumerate()
                                .map(|(i, x)| {
                                    x.map(|&epoch| {
                                        NaiveDateTime::from_timestamp(
                                            epoch,
                                            fractions.value(i) as u32,
                                        )
                                    })
                                })
                                .collect();
                            let series = Series::new(field.name.as_ref(), &datetimes);

                            found_series.append(&series).expect("TODO: panic message");
                        }
                        _ => unreachable!(),
                    }
                }

                _dt => {
                    // unreachable!("Unsupported data type {:?}", dt);
                }
            }
        }
    }

    return DataFrame::new(series_vec.values().cloned().collect());
}

pub fn get_chunks_for_data<R: Read>(
    arrow_data: &mut R,
) -> (
    Vec<Chunk<Box<dyn Array>>>,
    HashMap<usize, Metadata>,
    StreamMetadata,
) {
    let stream_metadata = read::read_stream_metadata(arrow_data).unwrap();

    let mut column_metadata: HashMap<usize, Metadata> = HashMap::new();

    // We need the field metadata for the timestamp info later.
    for (i, field) in stream_metadata.schema.fields.iter().enumerate() {
        column_metadata.insert(i, field.metadata.clone());
    }

    let stream = read::StreamReader::new(arrow_data, stream_metadata.clone());
    let mut chunks = vec![];

    for stream_state in stream {
        match stream_state {
            Ok(read::StreamState::Some(chunk)) => chunks.push(chunk),
            Ok(read::StreamState::Waiting) => break,
            Err(_l) => break,
        }
    }
    (chunks, column_metadata, stream_metadata)
}
