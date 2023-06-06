# DynamicLabelPropagation
Dynamic label propagation implemented in Rust.
Algorithm as described in the following paper:
Wang, Bo and Tu, Zhuowen and Tsotsos, John."Dynamic Label Propagation for Semi-supervised Multi-class Multi-label Classification." Proceedings of the ICCV, 2013.

If you clone the respository, you will have access to the test and training USPS digit data.

The function dynamic_label_propagation() takes number of classes, training labels/features, test features, as well as some tuning paramaters:

`dynamic_label_propagation(classes, &xTrain, &yTrain, &xTest, &Default::default())`

After compiling the source code, simply execute `cargo run` from the top level directory DynamicLabelPropagation in order to start the algorithm. The output will show the test array as well as the predictions:

![image](https://github.com/erkasc01/DynamicLabelPropagation/assets/50526911/48ce769d-7b21-4fcb-a1da-5f5847191366)
