mod neuron;
use neuron::{Neuron, NeuronOps};
use ndarray::array;

fn main() {
    let neu1 = Neuron {
        inputs: array![1.0, 2.0, 3.0, 2.5],
        weights: array![0.2, 0.8, -0.5, 1.0],
        bias: 2.0,
    };
    let neu2 = Neuron {
        inputs: array![1.0, 2.0, 3.0, 2.5],
        weights: array![0.5, -0.91, 0.26, -0.5],
        bias: 3.0,
    };
    let neu3 = Neuron {
        inputs: array![1.0, 2.0, 3.0, 2.5],
        weights: array![-0.26, -0.27, 0.17, 0.87],
        bias: 0.5,
    };
    let output1 = neu1.calculate_output();
    let output2 = neu2.calculate_output();
    let output3 = neu3.calculate_output();
    println!("{output1}, {output2}, {output3}");
}
