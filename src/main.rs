#![allow(non_snake_case)]
#![allow(unused_imports)]
use rand::Rng;
use std::fs::File;
use std::io;
use std::io::{BufRead,BufReader, Error, ErrorKind};
use smartcore::algorithm::neighbour::cover_tree::CoverTree;
use smartcore::math::distance::Distance;
use std::default::Default;
type FloatMat = Vec<Vec<f64>>;

pub struct Params
{
    sigma: f64,
    k: usize,
    lambda: f64,
    alpha: f64,
}
impl Default for Params
{
    fn default( ) -> Self
    {
        return Params{sigma: 0.6, k: 3, lambda: 0.10, alpha: 0.05}
    }
}
/*
pub fn read1(path: &str,delimiter:char) -> Result<FloatMat, Error>
{
    let file = File::open(path).expect("file not found");
    let br = BufReader::new(file);
    let mut v:FloatMat  = vec![];
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
pub fn USPSfeatures(file:&File) -> Result<FloatMat,Error>
{
    let mut features:FloatMat = vec![];
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
struct DistanceStruct<'a>
{
    graph: &'a FloatMat,
}


impl Distance<usize, f64> for DistanceStruct<'_>
{
    fn distance(&self, a:& usize, b:& usize) -> f64
    {
        self.graph[*a][*b]
    }
}

pub fn affinityMatrix(x:&FloatMat, sigma:f64)->FloatMat
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

pub fn dist_graph(mat:&FloatMat) -> FloatMat
{
    let n = mat.len();
    let mut g:FloatMat = vec![vec![0.;n];n];
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

pub fn calcSimMatrix(sampleMat:&FloatMat, params:&Params) -> (FloatMat,FloatMat)
{
    let num_samples = sampleMat.len();

    let mut ww = vec![vec![0.; num_samples]; num_samples];
    let affMat = affinityMatrix(&sampleMat,params.sigma);
    let g = dist_graph(&sampleMat);

    let ind:Vec<usize> =  (0..num_samples).collect();
    let tree = CoverTree::new(ind, DistanceStruct{graph: &g}).unwrap();

    for i in 0..num_samples
    {
        let knn = tree.find(&i, params.k).unwrap();
        for tup in knn
        {
            ww[i][*tup.2] = affMat[i][*tup.2];
        }
    }

    return (ww, affMat)
}

pub fn lambMat(num_samples:usize, params:&Params) -> FloatMat
{
    let mut mat = vec![vec![0.;num_samples];num_samples];
    for i in 0..num_samples
    {
        mat[i][i] = params.lambda;
    }
    return mat
}

pub fn probTransMatrix(sampleMat:&FloatMat,params:&Params)-> (FloatMat,FloatMat)
{
    let (w,ww) = calcSimMatrix(&sampleMat,&params);
    let n = w.len();
    let m = w[0].len();
    let mut p_0:FloatMat = vec![vec![0.;m];n];
    let mut ps:FloatMat = vec![vec![0.;m];n];

    for i in 0..n
    {
        for j in 0..m
        {
            p_0[i][j] = w[i][j];// initialize matrix identical to x
            ps[i][j] = ww[i][j];
        }
    }
    for i in 0..n
    {
        let rowSum1 = w[i].iter().sum::<f64>();
        let rowSum2 = ww[i].iter().sum::<f64>();

        for j in 0..m
        {
            p_0[i][j] /= rowSum1; //sum row and divide each element in the row by that value
            ps[i][j] /= rowSum2;
        }
    }
    return (p_0,ps)
}

//dynamic label propagation needs training data and test data to work on
//sigma is a tuning parameter for learning
pub fn dynamicLabelPropagation(trainFeatures:&FloatMat,trainLabels:&Vec<f64>,testFeatures:&FloatMat,testLabels:&Vec<f64>,num_samples:usize, params:&Params)->FloatMat
{
    let m = trainFeatures[0].len();
    let n = testFeatures.len();
    let mut trainFeatureSamples:FloatMat = vec![vec![0.;m];num_samples];
    let mut y:FloatMat = vec![vec![0.;10];num_samples+n];
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



    let (_p_0,ps) = probTransMatrix(&trainFeatureSamples,params);


    let lambdaMat = lambMat(num_samples, params);

    return ps
}


fn main()
{

    //load in training features and labels
    let file1 = File::open("TrainData/uspstrainlabels.txt").unwrap();
    let file2 = File::open("TrainData/uspstrainfeatures.txt").unwrap();
    let file3 = File::open("TestData/uspstestfeatures.txt").unwrap();
    let file4 = File::open("TestData/uspstestlabels.txt").unwrap();
    let trainLabels = USPSlabels(&file1).unwrap();
    let trainFeatures = USPSfeatures(&file2).unwrap();
    let testLabels = USPSlabels(&file4).unwrap();
    let testFeatures = USPSfeatures(&file3).unwrap();


    let test = dynamicLabelPropagation(&trainFeatures,&trainLabels,&testFeatures,&testLabels,5,&Default::default());
    for i in 0..5
    {
        println!("{:?}", test[i]);
    }
}
