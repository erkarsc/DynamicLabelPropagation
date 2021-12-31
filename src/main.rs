#![allow(non_snake_case)]
#![allow(unused_imports)]
use rand::Rng;
use std::fs::File;
use std::io;
use std::io::{BufRead,BufReader, Error, ErrorKind};
use smartcore::algorithm::neighbour::cover_tree::CoverTree;
use smartcore::math::distance::Distance;

/*
pub fn read1(path: &str,delimiter:char) -> Result<Vec<Vec<f64>>, Error>
{
    let file = File::open(path).expect("file not found");
    let br = BufReader::new(file);
    let mut v:Vec<Vec<f64>>  = vec![];
    for line in br.lines()
    {
        let mut pair:Vec<f64> = vec![];

        for x in line?.trim().split(delimiter)
        {
            let parsed:f64 = x.parse().unwrap();
            pair.push(parsed);
        }

        v.push(pair);
    }
    Ok(v)
}
*/

pub fn USPSlabels(file:&File)->Result<Vec<f64>,Error>
{

    let br = BufReader::new(file);
    let n:usize = 7291;
    let mut labels:Vec<f64> = vec![0.;n];
    let mut count:usize = 0;
    for line in br.lines()
    {
        //let temp = line.unwrap().chars().nth(0).unwrap().to_digit(10).unwrap() as f64 -1.;
        let temp  = line?.trim().parse().unwrap();
        labels[count] = temp;
        count +=1;

    }
    Ok(labels)
}
pub fn USPSfeatures(file:&File) -> Result<Vec<Vec<f64>>,Error>
{
    let mut features:Vec<Vec<f64>> = vec![];
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


pub fn norm(x:&Vec<f64>)->f64
{
    return x.iter().fold(0.,|sum,x|sum + x*x).sqrt();
}

#[derive(Clone)]
struct DistanceStruct
{
    graph: Vec<Vec<f64>>,
}

impl Distance<usize, f64> for DistanceStruct
{
    fn distance(&self, a: &usize, b: &usize) -> f64
    {
        self.graph[a][b]
    }
}

pub fn affinityMatrix(x:&Vec<Vec<f64>>, sigma:f64)->Vec<Vec<f64>>
{
    let n = x.len();
    let mut w = vec![vec![0.;n]; n]; // allocate space for affinity matrix
    let mut dif:Vec<f64>;
    let mut val:f64;
    for i in 0..n
    {
        for j in 0..n
        {
            dif = x[i].iter().zip(x[j].iter()).map(|(&x1,&x2)|x1-x2).collect();// compute difference of two rows
            val = norm(&dif).powf(2.)/sigma;
            //find norm of vector
            w[i][j] = (-val).exp();
        }
    }
    return w
}

pub fn probtransMatrix(x:&Vec<Vec<f64>>)-> Vec<Vec<f64>>
{
    let n = x.len();
    let m = x[0].len();
    let mut p:Vec<Vec<f64>> = vec![vec![0.;m];n];
    for i in 0..n
    {
        for j in 0..m
        {
            p[i][j] = x[i][j];// initialize matrix identical to x
        }
    }
    for i in 0..n
    {
        let rowSum = x[i].iter().sum::<f64>();
        for j in 0..m
        {
            p[i][j] /= rowSum; //sum row and divide each element in the row by that value
        }
    }
    return p
}




//computes weighted graph of k-nearest neighbors for points in matrix
pub fn kNN_graph(mat:&Vec<Vec<f64>>,_k:usize) -> Vec<Vec<f64>>
{
    let n = mat.len();
    let mut g:Vec<Vec<f64>> = vec![vec![0.;n];n];
    for i in 0..n
    {
        for j in 0..n
        {
            let temp = mat[i].iter().zip(mat[j].iter()).map(|(&mat1,&mat2)|mat1-mat2).collect();// compute norm of difference between rows
            g[i][j] = norm(&temp);
        }
    }

    return g
}

//dynamic label propagation needs training data and test data to work on
//sigma is a tuning parameter for learning
pub fn DynamicLabelPropagation(trainFeatures:&Vec<Vec<f64>>,trainLabels:&Vec<f64>,num_samples:usize, sigma:f64)->Vec<Vec<f64>>
{
    let m = trainFeatures[0].len();
    let mut trainFeatureSamples:Vec<Vec<f64>> = vec![vec![0.;m];num_samples];
    let mut y:Vec<Vec<f64>> = vec![vec![0.;10];num_samples];
    let mut rng = rand::thread_rng();
    for i in 0..num_samples
    {
        let randnum = rng.gen_range(0..trainFeatures.len());
        y[i][trainLabels[randnum] as usize] = 1.;
        for j in 0..m
        {
            trainFeatureSamples[i][j] = trainFeatures[randnum][j];
        }
    }

    let g = kNN_graph(&trainFeatureSamples,2);
    let w = affinityMatrix(&trainFeatureSamples,sigma);
    let ww = vec![vec![0.; num_samples]; num_samples];
    
    let mut tree = CoverTree::new(trainFeatureSamples, DistanceStruct{graph: g}).unwrap();
    let mut knn: Vec<usize>;
    for i in 0..num_samples
    {
        knn = tree.find(&i, 3/* <- k */).unwrap();
        for j in knn
        {
            ww[i][j] = w[i][j];
        }
    }

    let _p_0 = probtransMatrix(&w);

    return g
}

fn main()
{

    //load in training features and labels
    let file1 = File::open("TrainData/uspstrainlabels.txt").unwrap();
    let file2 = File::open("TrainData/uspstrainfeatures.txt").unwrap();
    let trainLabels = USPSlabels(&file1).unwrap();
    let trainFeatures = USPSfeatures(&file2).unwrap();
    let test = DynamicLabelPropagation(&trainFeatures,&trainLabels,5,0.6);
    for i in 0..5
    {
        println!("{:?}", test[i]);
    }
}
