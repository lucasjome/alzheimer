#include "itkRunLengthTextureFeaturesImageFilter.h"
#include "itkImage.h"
#include "itkVector.h"
#include "bipMiscellaneous.h"
#include "bipUtils.h"
#include "CmdLine.h"
#include "itkNeighborhood.h"

/*

Features extracted using this filter:
    1) Short run emphasis
    2) Long run emphasis
    3) Grey level non uniformity
    4) Run length non uniformity
    5) Low grey level run emphasis
    6) High grey level run emphasis
    7) Short run low grey level emphasis
    8) Short run high grey level emphasis
    9) Long run low grey level emphasis
   10) Long run high grey level emphasis

*/

void showUsage(std::string str)
{
	std::cout << "Usage:\n"
	<< "\t" << str << "\n"
	<< "\t-i inputImage\n"
	<< "\t-m maskImage [optional]\n"
	<< "\t-o outputImage\n"
	<< "\t-nb [number of bins per axis, default = 32]\n"
	<< "\t-pv [pixel value (min, max), default = (0, 255)]\n"
	<< "\t-dv [distance value (min, max), default = (1, 4)]\n"
	<< "\t-nr [neighborhood radius, default = 2]\n";
}



int main(int argc, char **argv)
{
	CCmdLine cmdLine;
	
	if(cmdLine.SplitLine(argc, argv) < 2
		|| (!cmdLine.HasSwitch("-i") || !cmdLine.HasSwitch("-o")))
	{
		showUsage(argv[0]);
		return -1;
	}

    // Setup types
    using InputImageType = itk::Image< int, 3 >;
    using OutputImageType = itk::Image< itk::Vector< float, 10 > , 3 >;
    using readerType = itk::ImageFileReader< InputImageType >;
    using NeighborhoodType = itk::Neighborhood< typename InputImageType::PixelType, InputImageType::ImageDimension >;
    NeighborhoodType neighborhood;

    // Read input image
    InputImageType::Pointer input_img = bip::utils::ReadImage< InputImageType >(cmdLine.GetArgument("-i", 0));

    // Read mask image (it must be const)
    //InputImageType::ConstPointer mask_img = cmdLine.HasSwitch("-m") ? (InputImageType::ConstPointer)bip::utils::ReadImage< InputImageType >(cmdLine.GetArgument("-m", 0)) : (InputImageType::ConstPointer)bip::misc::allocate_mask_image< InputImageType, InputImageType >(input_img);
    bip::misc::MaskImageType::ConstPointer mask_img = cmdLine.HasSwitch("-m") ? (bip::misc::MaskImageType::ConstPointer)bip::utils::ReadImage< bip::misc::MaskImageType >(cmdLine.GetArgument("-m", 0)) : (bip::misc::MaskImageType::ConstPointer)bip::misc::allocate_mask_image< InputImageType, bip::misc::MaskImageType >(input_img);
    
    // Set filter parameters and apply it
    using FilterType = itk::Statistics::RunLengthTextureFeaturesImageFilter< InputImageType, OutputImageType >;
    FilterType::Pointer filter = FilterType::New();
    filter->SetInput(input_img);
    filter->SetMaskImage(mask_img);
    filter->SetNumberOfBinsPerAxis(std::atoi(cmdLine.GetSafeArgument("-nb", 0, "32").c_str()));
    filter->SetHistogramValueMinimum(std::atof(cmdLine.GetSafeArgument("-pv", 0, "0").c_str()));
    filter->SetHistogramValueMaximum(std::atof(cmdLine.GetSafeArgument("-pv", 1, "255").c_str()));
    filter->SetHistogramDistanceMinimum(std::atof(cmdLine.GetSafeArgument("-dv", 0, "1").c_str()));
    filter->SetHistogramDistanceMaximum(std::atof(cmdLine.GetSafeArgument("-dv", 1, "4").c_str()));
    neighborhood.SetRadius(std::atoi(cmdLine.GetSafeArgument("-nr", 0, "2").c_str()));
    filter->SetNeighborhoodRadius(neighborhood.GetRadius());
    
    // Save output
    bip::utils::WriteImage< OutputImageType >(filter->GetOutput(), cmdLine.GetArgument("-o", 0));

  	return EXIT_SUCCESS;
}