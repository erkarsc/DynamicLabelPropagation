#![allow(non_snake_case)]
//extern crate blas_src;
use rand::Rng;
use std::fs::File;
use std::default::Default;
use std::io::{BufRead,BufReader, Error};
//use std::time::Instant;
use smartcore::algorithm::neighbour::cover_tree::CoverTree;
use smartcore::math::distance::Distance;
use ndarray::Data;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

type FloatMat = Vec<Vec<f64>>;


#[derive(Debug)]

//these are parameters used in the algorithm
//k is for k-nearest neighbors and sigma is for calculating the affinity matrix of the dataset
//alpha and lambda are paramaters in the final steps of the algorithm, but the results are not sensitive to either of them
pub struct Params
{
    k: usize,
    sigma: f64,
    alpha: f64,
    lambda: f64,
    max_iter: usize,
}

impl Default for Params
{
    fn default() -> Self
    {
        return Params{k:30, sigma:0.06, alpha:0.05, lambda:0.1, max_iter:50};
    }
}

pub fn USPSlabels(file:&File)->Result<Vec<f64>,Error>
{
    let br = BufReader::new(file);
    //let n:usize = 7291; //fix this
    let mut labels = vec![];
    for line in br.lines()
    {
        let temp  = line?.trim().parse().unwrap();
        labels.push(temp);
    }
    Ok(labels)
}

pub fn USPSfeatures(file:&File) -> Result<FloatMat,Error>
{
    let mut features = vec![];
    let br = BufReader::new(file);
    for line in br.lines()
    {
        let mut row = vec![];
        for x in line?.trim().split(',')
        {
            let parsed:f64 = x.parse().unwrap();
            row.push(parsed);
        }
        features.push(row);
    }
    Ok(features)
}

pub fn norm<S>(x:&ArrayBase<S, Ix1>)->f64
where
    S:Data<Elem = f64>,
{
    return x.iter().fold(0.,|sum,x|sum + x*x).sqrt();
}

pub fn accuracyScore(vec1:&Vec<usize>,vec2:&Vec<usize>)
{
    let mut count:f64 = 0.;

    for i in 0..vec2.len()
    {
        if vec1[i] == vec2[i]
        {
            count += 1.;
        }

    }
    let accuracyScore:f64 = count/(vec1.len() as f64);
    println!("{}{}", accuracyScore, "\n");
}

#[derive(Clone)]
struct DistanceStruct<'a>
{
    graph: &'a Array2<f64>,
}

impl<'a> Distance<usize, f64> for DistanceStruct<'a>
{
    fn distance(&self, a:& usize, b:& usize) -> f64
    {
        self.graph[[*a,*b]]
    }
}

pub fn dist_graph(mat:&Array2<f64>) -> Array2<f64>
{
    let n = mat.shape()[0];
    let mut g = Array2::<f64>::zeros((n,n));//vec![vec![0.;n];n];
    for i in 0..n
    {
        for j in 0..n
        {
            let temp = &mat.row(i)-&mat.row(j);
            g[[i,j]] = norm(&temp);
        }
    }
    return g
}


//creates affinity matrix of a given graph
pub fn affinityMatrix(x:&Array2<f64>, params: &Params)->Array2<f64>
{
    let n = x.shape()[0];
    let mut w = Array2::<f64>::zeros((n,n));//vec![vec![0.;n]; n]; // allocate space for affinity matrix
    let mut val:f64;
    for i in 0..n
    {
        for j in 0..n
        {
            let dif = &x.row(i)-&x.row(j);// compute difference of two rows
            val = norm(&dif).powf(2.)/params.sigma;
            //find norm of vector
            w[[i,j]] = (-val).exp();

        }
    }
    return w
}

// calculates the similarity matrices that will be used to obtain the probabilistic transition matrices
pub fn calcSimMatrix(sampleMat:&Array2<f64>, params:&Params) -> (Array2<f64>,Array2<f64>)
{
    let num_samples = sampleMat.shape()[0];

    let mut ww = Array2::<f64>::zeros((num_samples,num_samples));//vec![vec![0.; num_samples]; num_samples];
    let affMat = affinityMatrix(&sampleMat,params);
    let g = dist_graph(&sampleMat);

    let ind:Vec<usize> =  (0..num_samples).collect();
    let tree = CoverTree::new(ind, DistanceStruct{graph: &g}).unwrap();

    for i in 0..num_samples
    {
        let knn = tree.find(&i, params.k).unwrap();
        for tup in knn
        {
            ww[[i,*tup.2]] = affMat[[i,*tup.2]];
        }
    }
    return (ww, affMat)
}

//creates probabilistic transition matrices for W and 𝓦
pub fn probTransMatrix(sampleMat:&Array2<f64>,params:&Params)-> (Array2<f64>,Array2<f64>)
{
    let (w,ww) = calcSimMatrix(&sampleMat,&params);
    let n = w.shape()[0];
    let m = w.len_of(Axis(0));
    let mut p_0 = Array2::<f64>::zeros((n,m));//vec![vec![0.;m];n];
    let mut ps = Array2::<f64>::zeros((n,m));//vec![vec![0.;m];n];

    for i in 0..n
    {
        for j in 0..m
        {
            p_0[[i,j]] = w[[i,j]];// initialize matrix identical to x
            ps[[i,j]] = ww[[i,j]];
        }
    }

    for i in 0..n
    {
        let rowSum1:f64 = w.row(i).sum();
        let rowSum2:f64 = ww.row(i).sum();
        for j in 0..m
        {
            p_0[[i,j]] /= rowSum1; //sum row and divide each element in the row by that value
            ps[[i,j]] /= rowSum2;
        }
    }
    return (p_0,ps)
}
//lambda matrix used in final iteration steps of algorithm
//although lambda is a parameter, the algorithm is not very sensitive to changes in it
pub fn lambMat(num_samples:usize, params:&Params) -> Array2<f64>
{
    let n = 100;// This is the number of unlabeled samples
    let mut mat = Array2::<f64>::zeros((num_samples+n,num_samples+n));//vec![vec![0.;num_samples+n];num_samples+n];
    for i in 0..num_samples
    {
        mat[[i,i]] = params.lambda;
    }
    return mat
}

pub fn labelMat(numClasses:usize, labeledFeatures:&FloatMat, labels:&Vec<f64>, unlabeledFeatures:&FloatMat, num_samples:usize) -> (Array2<f64>,Array2<f64>)
{
    let m = labeledFeatures[0].len();
    let n = 100; // This is the number of unlabeled samples
    let mut featureSamples = Array2::<f64>::zeros((num_samples+n,m));
    let mut y = Array2::<f64>::zeros((num_samples+n,numClasses));
    let mut rng = rand::thread_rng();
    for i in 0..num_samples + n
    {
        if i < num_samples
        {
            let randnum = rng.gen_range(0..labeledFeatures.len());
            y[[i,labels[randnum] as usize]] = 1.;
            for j in 0..m
            {
                featureSamples[[i,j]] = labeledFeatures[randnum][j];
            }
        }
        else
        {
            let randnum = rng.gen_range(0..unlabeledFeatures.len());
            for j in 0..m
            {
                featureSamples[[i,j]] = unlabeledFeatures[randnum][j];
            }
        }
    }
    return (y,featureSamples)
}

//dynamic label propagation needs training data and test data to work on
//sigma is a tuning parameter
pub fn dynamicLabelPropagation(numClasses:usize,labeledFeatures:&FloatMat,labels:&Vec<f64>,unlabeledFeatures:&FloatMat,num_samples:usize, params:&Params)->(Array2<f64>,Vec<usize>)
{
    let(y,featureSamples) = labelMat(numClasses,&labeledFeatures, &labels, &unlabeledFeatures, num_samples);

    let (mut p_0,ps) = probTransMatrix(&featureSamples,params);

    let lambdaMat = lambMat(num_samples, params);

    let mut yNew = Array2::<f64>::zeros((p_0.shape()[0],y.shape()[1]));

    let psT = ps.t();

    for _i in 0..params.max_iter
    {
        yNew = p_0.dot(&y);

        for i in 0..num_samples
        {
            for j in 0..yNew.shape()[1]
            {
                yNew[[i,j]] = y[[i,j]];
            }
        }
        p_0 = &ps.dot(&(&p_0 + params.alpha*y.dot(&y.t()))).dot(&psT) + &lambdaMat;
        //p_0 = matSum(&matMult(&matMult(&ps,&matSum(&p_0,&scalarMult(params.alpha,matMult(&y,&y.t())))),&ps.t()),&lambdaMat);
    }

    let mut predictedLabels = vec![];

    for i in num_samples..yNew.shape()[0]
    {
        predictedLabels.push(yNew.row(i).argmax().unwrap());
    }

    return (yNew,predictedLabels)
}

fn main()
{

    /*
    let file1 = File::open("./TrainData/uspstrainlabels.txt").unwrap();
    let file2 = File::open("./TrainData/uspstrainfeatures.txt").unwrap();
    let file3 = File::open("./TestData/uspstestfeatures.txt").unwrap();
    let file4 = File::open("./TestData/uspstestlabels.txt").unwrap();
*/
    let file5 = File::open("./WineData/redwinelabels.txt").unwrap();
    let file6 = File::open("./WineData/redwinefeatures.txt").unwrap();

    let labelData = USPSlabels(&file5).unwrap();
    let featureData = USPSfeatures(&file6).unwrap();


    let trainNum = 500;
    let n = labelData.len();


    let mut xTrain:FloatMat = vec![vec![0.;11];trainNum];
    let mut yTrain = vec![0.; trainNum];
    for i in 0..trainNum
    {
        for j in 0..11
        {
            xTrain[i][j] = featureData[i][j];
        }
        yTrain[i] = labelData[i];
    }

    let mut xTest:FloatMat = vec![vec![0.;11]; n-trainNum];
    let mut yTest = vec![0.; n-trainNum];

    for i in trainNum..n
    {
        for j in 0..11
        {
            xTest[i-trainNum][j] = featureData[i][j];
        }
        yTest[i-trainNum] = labelData[i];
    }

/*
    let testLabels = USPSlabels(&file4).unwrap();
    let testFeatures = USPSfeatures(&file3).unwrap();
*/

    let classes:usize = 11;
    let _test = dynamicLabelPropagation(classes,&xTrain,&yTrain,&xTest,500,&Default::default());



}
