#![allow(non_snake_case)]
#![allow(unused_imports)]

use rand::Rng;

use std::fs::File;
use std::io;
use std::default::Default;
use std::io::{BufRead,BufReader, Error, ErrorKind};

use smartcore::algorithm::neighbour::cover_tree::CoverTree;
use smartcore::math::distance::Distance;

type FloatMat = Vec<Vec<f64>>;

pub fn printMat(mat: &FloatMat)
{
    print!("[");

    for i in 0..(mat.len())
    {
        if i != 0
        {
            print!(" ");
        }

        print!("[");
        for j in 0..(mat[i].len())
        {
            print!("{:.3}", mat[i][j]);
            if j != (mat[i].len()-1)
            {
                print!(", ");
            }
        }
        print!("]");
        if i != (mat.len()-1)
        {
            print!("\n");
        }
    }

    println!("]");
}

pub fn matMul(mat1:FloatMat, mat2:FloatMat) -> FloatMat
{
    let r1 = mat1.len();
    let r2 = mat2.len();
    let c1 = mat1[0].len();
    let c2 = mat2[0].len();
    let mut prod = vec![vec![0.;c2];r1];
    for i in 0..r1
    {
        for j in 0..c2
        {

            for k in 0..r2
            {
              prod[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return prod
}

#[derive(Debug)]
pub struct Params
{
    k: usize,
    sigma: f64,
    alpha: f64,
    lambda: f64,
}

impl Default for Params
{
    fn default() -> Self
    {
        return Params{k:3, sigma:0.6, alpha:0.05, lambda:0.1};
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

impl<'a> Distance<usize, f64> for DistanceStruct<'a>
{
    fn distance(&self, a:& usize, b:& usize) -> f64
    {
        self.graph[*a][*b]
    }
}

//creates affinity matrix of a given graph
pub fn affinityMatrix(x:&FloatMat, params: &Params)->FloatMat
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
            val = norm(&dif).powf(2.)/params.sigma;
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

// calculates the similarity matrices that will be used to obtain the probabilistic transition matrices
pub fn calcSimMatrix(sampleMat:&FloatMat, params:&Params) -> (FloatMat,FloatMat)
{
    let num_samples = sampleMat.len();

    let mut ww = vec![vec![0.; num_samples]; num_samples];
    let affMat = affinityMatrix(&sampleMat,params);
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


//creates probabilistic transition matrices for W and 𝓦
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
//lambda matrix used in final iteration steps of algorithm
//although lambda is a parameter, the algorithm is not very sensitive to changes in it
pub fn lambMat(num_samples:usize, params:&Params) -> FloatMat
{
    let mut mat = vec![vec![0.;num_samples];num_samples];
    for i in 0..num_samples
    {
        mat[i][i] = params.lambda;
    }
    return mat
}

pub fn labelMat(labeledFeatures:&FloatMat, labels:&Vec<f64>, unlabeledFeatures:&FloatMat,num_samples:usize) -> (FloatMat,FloatMat)
{
    let m = labeledFeatures[0].len();
    let n = unlabeledFeatures.len();
    let mut labeledFeatureSamples:FloatMat = vec![vec![0.;m];num_samples];
    let mut y:FloatMat = vec![vec![0.;10];num_samples+n];
    let mut rng = rand::thread_rng();
    for i in 0..num_samples
    {
        let randnum = rng.gen_range(0..labeledFeatures.len());
        y[i][labels[randnum] as usize] = 1.;
        for j in 0..m
        {
            labeledFeatureSamples[i][j] = labeledFeatures[randnum][j];
        }
    }
    return (y,labeledFeatureSamples)
}

//dynamic label propagation needs training data and test data to work on
//sigma is a tuning parameter
pub fn dynamicLabelPropagation(labeledFeatures:&FloatMat,labels:&Vec<f64>,unlabeledFeatures:&FloatMat,testLabels:&Vec<f64>,num_samples:usize, params:&Params)->FloatMat
{

    let(y,labeledFeatureSamples) = labelMat(&labeledFeatures, &labels, &unlabeledFeatures, num_samples);

    let (p_0,_ps) = probTransMatrix(&labeledFeatureSamples,params);

    let lambdaMat = lambMat(num_samples, params);

    return p_0
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


    let _test = dynamicLabelPropagation(&trainFeatures,&trainLabels,&testFeatures,&testLabels,10,&Default::default());

    //printMat(&test);
}
