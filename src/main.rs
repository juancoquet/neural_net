use ndarray::array;

fn main() {
    // batch of 3x input samples n=4
    let inputs = array![
        [1.0, 2.0, 3.0, 2.5],   // batch 1
        [2.0, 5.0, -1.0, 2.0],  // batch 2
        [-1.5, 2.7, 3.3, -0.8]  // batch 3
    ];

    // weights from each input to each neuron
    let weights1 = array![
        [0.2, 0.8, -0.5, 1.0],      // weights for n1.1
        [0.5, -0.91, 0.26, -0.5],   // weights for n1.2
        [-0.26, -0.27, 0.17, 0.87]  // weights for n1.3
    ];
    let biases1 = array![2.0, 3.0, 0.5];
    // 3 x 3 matrix - (row = batch, col = neuron)
    let layer1_outputs = inputs.dot(&weights1.t()) + biases1;

    // weights from first layer outputs to second layer
    let weights2 = array![
        [0.1, -0.14, 0.5],    // weights for n2.1
        [-0.5, 0.12, -0.33],  // weights for n2.2
        [-0.44, 0.73, -0.13]  // weights for n2.3
    ];
    let biases2 = array![-1., 2., -0.5];
    // 3 x 3 matrix - (row = batch, col = neuron)
    let layer2_outpus = layer1_outputs.dot(&weights2.t()) + biases2;
    println!("{layer2_outpus}")
}
