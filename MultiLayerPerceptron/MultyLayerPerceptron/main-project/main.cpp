#include "../basic-includes.h"

#include "../02-Dense-Layer/DenseLayer.h"
#include "../03-Activation-Function/ActivationFunction.h"
#include "../04-MLP/MLP.h"
#include "DataLoader.h"
#include "../08-Optimizers/Optimizers.h"
#include "comparations.h"

using MNIST_DATA = std::vector<std::pair<Eigen::MatrixXd, size_t>>;



int main(int argc, const char** argv)
{
	GnuplotWrap gnuplot;
	gnuplot.OutFile("..\\..\\res.dat");
	gnuplot.xRange("0", "");
	gnuplot.yRange("-0.01", "1.05");
	gnuplot.Grid("1", "0.1");

	//"..\\..\\MNIST_dataset\\train"
	MNIST_DATA traingDatas = DataLoader("D:\\w\\.debugger\\GITHUB\\TCC-repositories\\Multi-Layer-Perceptrom\\MNIST_dataset\\train").Load();
	MNIST_DATA testDatas = DataLoader("D:\\w\\.debugger\\GITHUB\\TCC-repositories\\Multi-Layer-Perceptrom\\MNIST_dataset\\test").Load();


	MLP mlp = MLPbuilder()
				.InputSize(28*28)
				.BatchSize(64)
				.Architecture({
					new DenseLayer(128, 0.0001),
					new ReLU(),
					new Dropout(0.3),

					new DenseLayer(64, 0.0001),
					new ReLU(),
					new Dropout(0.1),

					new DenseLayer(10, 0.0001),
				})
				.LossFunction(new SoftmaxEntropy)
				.MaxEpochs(30)
				.UseAdam(0.9)
				.Build();


	size_t epoch = 0;

	mlp.Training(traingDatas, [&]() {
		std::cout << "\n\n\n---------------------------------------------- " << epoch << " ----------------------------------------------\n\n\n\n";

		std::cout << "TRAINING:\n";
		double trainAccuracy = Evaluator::Eval_MLP(mlp, traingDatas);

		std::cout << "\n\nTEST:\n";
		double testAccuracy = Evaluator::Eval_MLP(mlp, testDatas);

		gnuplot.out << epoch << " " << trainAccuracy << " " << testAccuracy << "\n";

		epoch++;
	});


	gnuplot.out.close();
	gnuplot << "plot \'..\\..\\res.dat\' using 1:2 w l title \"Training error\", ";
	gnuplot << "\'..\\..\\res.dat\' using 1:3 w l title \"Test error\" \n";
	gnuplot << " \n";


	std::cout << "\n\n\n[SUCCESS]!!!!!\n";
	return 0;
}



//-------------------------------------------
// conparations
//-------------------------------------------

int __main(int argc, const char** argv)
{
	MNIST_DATA traingDatas;
	//MNIST_DATA traingDatas = DataLoader("D:\\w\\.debugger\\GITHUB\\TCC-repositories\\Multi-Layer-Perceptrom\\MNIST_dataset\\train").Load();
	//MNIST_DATA testDatas = DataLoader("D:\\w\\.debugger\\GITHUB\\TCC-repositories\\Multi-Layer-Perceptrom\\MNIST_dataset\\test").Load();
	
	//Comparations::With_WithOut_Adam(traingDatas, testDatas); 
	//Comparations::BatchSize(traingDatas, testDatas);
	//Comparations::LossFunction(traingDatas, testDatas);
	//Comparations::ActivationFunction(traingDatas, testDatas);
	Comparations::Adam_beta(traingDatas, traingDatas);

	return 0;
}

