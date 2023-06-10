mod data_files;
mod layers;

use data_files::data_loader::load_data;

fn main() {
    let data = load_data("src/data_files/spiral_data.json").unwrap();
    let inputs = data.input_batch;

    let n_inputs = 2;
    let n_neurons = 3;
    let dense1 = layers::LayerDense::new(n_inputs, n_neurons);
    let output1 = dense1.forward(inputs);
    println!("{output1}")
}
