use ndarray::{Array1, Array2};
use serde::{Deserialize, Deserializer};
use std::fs;

#[derive(Deserialize)]
pub struct Data {
    #[serde(deserialize_with = "deserialize_array2")]
    pub x: Array2<f64>,
    #[serde(deserialize_with = "deserialize_array1")]
    pub y: Array1<f64>,
}

fn deserialize_array2<'de, D>(deserializer: D) -> Result<Array2<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let data: Vec<Vec<f64>> = Deserialize::deserialize(deserializer)?;
    let num_rows = data.len();
    let num_cols = data[0].len();

    let flat_data: Vec<f64> = data.into_iter().flatten().collect();
    Ok(Array2::from_shape_vec((num_rows, num_cols), flat_data).unwrap())
}

fn deserialize_array1<'de, D>(deserializer: D) -> Result<Array1<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let data: Vec<f64> = Deserialize::deserialize(deserializer)?;
    Ok(Array1::from_vec(data))
}

pub fn load_data(file_path: &str) -> Result<Data, Box<dyn std::error::Error>> {
    let json_data = fs::read_to_string(file_path)?;
    let data: Data = serde_json::from_str(&json_data)?;
    Ok(data)
}
