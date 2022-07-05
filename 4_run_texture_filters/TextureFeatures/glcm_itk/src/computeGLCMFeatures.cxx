#include <itkCastImageFilter.h>
#include "itkCoocurrenceTextureFeaturesImageFilter.h"
#include "itkImage.h"
#include "itkNthElementImageAdaptor.h"
#include "itkVector.h"
#include "bipMiscellaneous.h"
#include "bipUtils.h"
#include "CmdLine.h"
#include "itkNeighborhood.h"

/*

Features extracted using this filter:
    1) Energy
    2) Entropy
    3) Correlation
    4) Inverse Difference Moment
    5) Inertia
    6) Cluster shade
    7) Cluster prominence
    8) Haralick correlation

*/

void showUsage(std::string str)
{
    std::cout << "Usage:\n"
              << "\t" << str << "\n"
              << "\t-i inputImage\n"
              << "\t-m maskImage [optional]\n"
              << "\t-o outputImage\n"
              << "\t-nb      [number of bins per axis, default = 32]\n"
              << "\t-pv      [pixel value (min, max), default = (0, 255)]\n"
              << "\t-nr      [neighborhood radius, default = 2]\n"
              << "\t-sepfeat [separate features into different images, default = one vector image with all features]\n";
}

std::string remove_extension(std::string filename)
{
    size_t last_idx = filename.find_last_of(".");
    return last_idx == std::string::npos ? filename : filename.substr(0, last_idx);
}

int main(int argc, char **argv)
{
    CCmdLine cmdLine;

    if (cmdLine.SplitLine(argc, argv) < 2 || (!cmdLine.HasSwitch("-i") || !cmdLine.HasSwitch("-o")))
    {
        showUsage(argv[0]);
        return -1;
    }

    bool separate = cmdLine.HasSwitch("-sepfeat");
    std::string output_string = remove_extension(cmdLine.GetArgument("-o", 0));

    // Constants
    const int NumberOfFeatures = 8;
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

    // Set filter parameters and apply it
    using FilterType = itk::Statistics::CoocurrenceTextureFeaturesImageFilter<InputImageType, OutputImageType>;
    FilterType::Pointer filter = FilterType::New();
    filter->SetInput(input_img);
    filter->SetInsidePixelValue(1);
    filter->SetMaskImage(mask_img);
    filter->SetNumberOfBinsPerAxis(std::atoi(cmdLine.GetSafeArgument("-nb", 0, "32").c_str()));
    filter->SetHistogramMinimum(std::atof(cmdLine.GetSafeArgument("-pv", 0, "0").c_str()));
    filter->SetHistogramMaximum(std::atof(cmdLine.GetSafeArgument("-pv", 1, "255").c_str()));
    neighborhood.SetRadius(std::atoi(cmdLine.GetSafeArgument("-nr", 0, "2").c_str()));
    filter->SetNeighborhoodRadius(neighborhood.GetRadius());

    std::string sep_filename_preamble = remove_extension(output_string);

    if (separate)
    {
        typedef itk::NthElementImageAdaptor<OutputImageType, float> ImageAdaptorType;
        ImageAdaptorType::Pointer adaptor = ImageAdaptorType::New();

        typedef itk::CastImageFilter<ImageAdaptorType, bip::misc::InputImageType> CastImageFilterType;
        CastImageFilterType::Pointer cast = CastImageFilterType::New();

        char buffer[400];

        for (int i = 0; i < NumberOfFeatures; ++i)
        {
            // Compress each feature map
            sprintf(buffer, "%s_glcm_feature_%d.nii.gz", sep_filename_preamble.c_str(), i + 1);

            adaptor->SelectNthElement(i);
            adaptor->SetImage(filter->GetOutput());

            cast->SetInput(adaptor);
            cast->Update();

            bip::utils::WriteImage<bip::misc::InputImageType>(cast->GetOutput(), buffer);
        }
    }
    else
    {
        // Save output
        bip::utils::WriteImage<OutputImageType>(filter->GetOutput(), output_string + ".nii.gz");
    }

    std::ofstream params;
    params.open("GLCM_parameters.txt");
    params << "\t\tGLCM PARAMETERS\n-----------------------------\n";
    params << "\tInput image: " << cmdLine.GetArgument("-i", 0) << "\n";
    params << (cmdLine.HasSwitch("-m") ? "\tMask: Yes\n" : "\tMask: No\n");
    params << "\tNumber of bins: " << std::atoi(cmdLine.GetSafeArgument("-nb", 0, "32").c_str()) << "\n";
    params << "\tHist min: " << std::atof(cmdLine.GetSafeArgument("-pv", 0, "0").c_str()) << "\n";
    params << "\tHist max: " << std::atof(cmdLine.GetSafeArgument("-pv", 1, "255").c_str()) << "\n";
    params << "\tRadius: " << std::atoi(cmdLine.GetSafeArgument("-nr", 0, "2").c_str()) << "\n";
    params << (separate ? "\tSeparate features: Yes\n" : "\tSeparate features: No\n");
    params.close();

    return EXIT_SUCCESS;
}