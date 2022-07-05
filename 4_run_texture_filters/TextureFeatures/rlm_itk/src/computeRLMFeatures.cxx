#include <itkCastImageFilter.h>
#include <itkImage.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkNeighborhood.h>
#include <itkNthElementImageAdaptor.h>
#include <itkRunLengthTextureFeaturesImageFilter.h>
#include <itkVector.h>

#include "CmdLine.h"

#include "bipMiscellaneous.h"
#include "bipUtils.h"

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

std::string remove_extension(std::string filename)
{
    size_t last_dot = filename.find_last_of(".");
    return last_dot == std::string::npos       //npos is reached when find_last_of() doesnt find the symbol
               ? filename                      // if filename has no extension, return itself
               : filename.substr(0, last_dot); // return filename without extesion
}

void showUsage(std::string str)
{
    std::cout << "Usage:\n"
              << "\t" << str << "\n"
              << "\t-i inputImage\n"
              << "\t-m maskImage [optional]\n"
              << "\t-o outputImage\n"
              << "\t-nb 	   [number of bins per axis, default = 32]\n"
              << "\t-pv 	   [pixel value (min, max), default = (min vxl value in img, max vxl value in img)]\n"
              << "\t-dv 	   [distance value (min, max), default = (1, 4)]\n"
              << "\t-nr 	   [neighborhood radius, default = 2]\n"
              << "\t-sepfeat [separate features into multiple images, default = one vector image with all features]\n";
}

void computeRLMTextureMaps(CCmdLine cmdLine)
{
    bool separate = cmdLine.HasSwitch("-sepfeat");
    std::string output_string = remove_extension(cmdLine.GetArgument("-o", 0));

    // Constants across this code
    const int NumberOfFeatures = 10;
    const int Dimension = 3;

    // Setup types
    using InputImageType = itk::Image<int, Dimension>;
    using OutputImageType = itk::Image<itk::Vector<float, NumberOfFeatures>, Dimension>;
    using readerType = itk::ImageFileReader<InputImageType>;
    using NeighborhoodType = itk::Neighborhood<typename InputImageType::PixelType, InputImageType::ImageDimension>;
    NeighborhoodType neighborhood;

    // Read input image
    InputImageType::Pointer input_img = bip::utils::ReadImage<InputImageType>(cmdLine.GetArgument("-i", 0));

    // Read mask image (it must be const)
    bip::misc::MaskImageType::ConstPointer mask_img = cmdLine.HasSwitch("-m") ? (bip::misc::MaskImageType::ConstPointer)bip::utils::ReadImage<bip::misc::MaskImageType>(cmdLine.GetArgument("-m", 0)) : (bip::misc::MaskImageType::ConstPointer)bip::misc::allocate_mask_image<InputImageType, bip::misc::MaskImageType>(input_img);

    // Get parameters that will be on the output filename
    int num_bins = std::atoi(cmdLine.GetSafeArgument("-nb", 0, "32").c_str());
    double dist_min = std::atof(cmdLine.GetSafeArgument("-dv", 0, "1").c_str());
    double dist_max = std::atof(cmdLine.GetSafeArgument("-dv", 1, "4").c_str());
    int radius = std::atoi(cmdLine.GetSafeArgument("-nr", 0, "2").c_str());

    // Compute min and max voxel values in the input image
    typedef itk::MinimumMaximumImageCalculator<InputImageType> ImageCalculatorFilterType;
    ImageCalculatorFilterType::Pointer calculator = ImageCalculatorFilterType::New();
    calculator->SetImage(input_img);
    calculator->Compute();

    // Set filter parameters and apply it
    using FilterType = itk::Statistics::RunLengthTextureFeaturesImageFilter<InputImageType, OutputImageType>;
    FilterType::Pointer filter = FilterType::New();
    filter->SetInput(input_img);
    filter->SetMaskImage(mask_img);
    filter->SetNumberOfBinsPerAxis(num_bins);
    filter->SetHistogramValueMinimum(cmdLine.HasSwitch("-pv") ? std::atof(cmdLine.GetArgument("-pv", 0).c_str()) : calculator->GetMinimum());
    filter->SetHistogramValueMaximum(cmdLine.HasSwitch("-pv") ? std::atof(cmdLine.GetArgument("-pv", 1).c_str()) : calculator->GetMaximum());
    filter->SetHistogramDistanceMinimum(dist_min);
    filter->SetHistogramDistanceMaximum(dist_max);
    neighborhood.SetRadius(radius);
    filter->SetNeighborhoodRadius(neighborhood.GetRadius());

    // Save output (vector image)

    //char parameters[50]; // buffer to hold the parameters using during processing
    //sprintf(parameters, "_nb-%d_distmin-%.2lf_distmax-%.2lf_radius-%d.nii", num_bins, dist_min, dist_max, radius);

    std::string sep_filename_preamble = remove_extension(output_string);

    if (separate)
    {
        typedef itk::NthElementImageAdaptor<OutputImageType, float> ImageAdaptorType;
        ImageAdaptorType::Pointer adaptor = ImageAdaptorType::New();

        // Cast filter to save image in an appropriate type (bip::misc::InputImageType is a double precision image)
        typedef itk::CastImageFilter<ImageAdaptorType, bip::misc::InputImageType> CastImageFilterType;
        CastImageFilterType::Pointer cast = CastImageFilterType::New();

        // Just a simple buffer to append feature number
        char buffer[400];

        for (int i = 0; i < NumberOfFeatures; ++i)
        {
            sprintf(buffer, "%s_rlm_feature_%d.nii.gz", sep_filename_preamble.c_str(), i + 1);

            // Adaptor to select one feature from the vector image
            adaptor->SelectNthElement(i);
            adaptor->SetImage(filter->GetOutput());

            // Cast it
            cast->SetInput(adaptor);
            cast->Update();

            // Save it
            bip::utils::WriteImage<bip::misc::InputImageType>(cast->GetOutput(), buffer);
        }
    }
    else
    {
        bip::utils::WriteImage<OutputImageType>(filter->GetOutput(), output_string + ".nii.gz");
    }
}

int main(int argc, char **argv)
{
    CCmdLine cmdLine;

    if (cmdLine.SplitLine(argc, argv) < 2 || (!cmdLine.HasSwitch("-i") || !cmdLine.HasSwitch("-o")))
    {
        showUsage(argv[0]);
        return -1;
    }

    computeRLMTextureMaps(cmdLine);

    return EXIT_SUCCESS;
}
