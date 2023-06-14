mod data_files;
mod layers;

use ndarray::s;

use data_files::data_loader::load_data;
use layers::{LayerDense, ActivationReLU};

fn main() {
    let data = load_data("src/data_files/spiral_data.json").unwrap();
    let inputs = data.input_batch;

    let n_inputs = 2; // per batch
    let n_neurons = 3;
    let dense1 = LayerDense::new(n_inputs, n_neurons);

    // matrix of n_inputs rows x n_neurons cols
    // each row is a processed input sample
    // each col is a neuron's output for the sample
    let dense1_output = dense1.forward(inputs);

    let activation1 = ActivationReLU{};
    let activation1_output = activation1.forward(dense1_output);

    let five = activation1_output.slice(s![0..5, ..]);
    println!("{five}")
}
