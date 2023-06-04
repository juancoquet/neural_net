mod neuron;
use neuron::{Neuron, get_output};

fn main() {
    let neu = Neuron {
        inputs: vec![1.0, 2.0, 3.0, 2.5],
        weights: vec![0.2, 0.8, -0.5, 1.0],
        bias: 2.0,
    };
    let output = get_output(neu);
    println!("{output}");
}
