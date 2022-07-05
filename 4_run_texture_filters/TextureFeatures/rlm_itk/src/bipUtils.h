#ifndef __bipUtils_h
#define __bipUtils_h

#include <iostream>
#include <vector>

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImage.h"
#include "itkImageDuplicator.h"
#include "itkGDCMImageIO.h"
#include "itkImageIOBase.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkOrientImageFilter.h"
#include "itkSpatialOrientation.h"
#include "itkSpatialOrientationAdapter.h"
#include "itkResampleImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkLabelImageGaussianInterpolateImageFunction.h"

#define LINEAR_INTERPOLATION 			0
#define BSPLINE_INTERPOLATION			1
#define NEAREST_NEIGHBOR_INTERPOLATION 	2
#define MULTI_LABEL_INTERPOLATION 3

namespace bip
{

namespace utils
{

typedef itk::SpatialOrientationAdapter SOAdapterType;
typedef SOAdapterType::DirectionType   DirectionType;

/**
  * This function reads an image using ITK -- image-based template
*/
template< typename ImageType >
typename ImageType::Pointer ReadImage( const std::string &fileName )
{
    typename ImageType::Pointer image;
    std::string extension = itksys::SystemTools::GetFilenameLastExtension( fileName );
    itk::GDCMImageIO::Pointer dicomIO = itk::GDCMImageIO::New();
    if( dicomIO->CanReadFile( fileName.c_str() ) || ( itksys::SystemTools::LowerCase(extension) == ".dcm" ) )
    {
        std::string dicomDir = itksys::SystemTools::GetParentDirectory( fileName.c_str() );

        itk::GDCMSeriesFileNames::Pointer FileNameGenerator = itk::GDCMSeriesFileNames::New();
        FileNameGenerator->SetUseSeriesDetails( true );
        FileNameGenerator->SetDirectory( dicomDir );

        typedef const std::vector< std::string > ContainerType;
        const ContainerType & seriesUIDs = FileNameGenerator->GetSeriesUIDs();

        typedef typename itk::ImageSeriesReader< ImageType > ReaderType;
        typename ReaderType::Pointer reader = ReaderType::New();
        reader->SetFileNames( FileNameGenerator->GetFileNames( seriesUIDs[0] ) );
        reader->SetImageIO( dicomIO );
        try
        {
            reader->Update();
        }
        catch( itk::ExceptionObject & err )
        {
            std::cout << "Caught an exception: " << std::endl;
            std::cout << err << " " << __FILE__ << " " << __LINE__ << std::endl;
            throw;
        }
        catch( ... )
        {
            std::cout << "Error while reading in image for patient " << fileName << std::endl;
            throw;
        }
        image = reader->GetOutput();
        image->DisconnectPipeline();
        //image->ReleaseDataFlagOn();
    }
    else
    {
        typedef itk::ImageFileReader< ImageType > ReaderType;
        typename ReaderType::Pointer reader = ReaderType::New();
        reader->SetFileName( fileName.c_str() );
        try
        {
            reader->Update();
        }
        catch( itk::ExceptionObject & err )
        {
            std::cout << "Caught an exception: " << std::endl;
            std::cout << err << " " << __FILE__ << " " << __LINE__ << std::endl;
            throw;
        }
        catch( ... )
        {
            std::cout << "Error while reading in image" << fileName << std::endl;
            throw;
        }
        image = reader->GetOutput();
        image->DisconnectPipeline();
        //image->ReleaseDataFlagOn();
    }
    return image;
}

/**
  * This function reads a constant image using ITK -- image-based template
*/
template< typename ImageType >
typename ImageType::ConstPointer ReadConstImage( const std::string &fileName )
{
    typename ImageType::ConstPointer image;
    std::string extension = itksys::SystemTools::GetFilenameLastExtension( fileName );
    itk::GDCMImageIO::Pointer dicomIO = itk::GDCMImageIO::New();
    if( dicomIO->CanReadFile( fileName.c_str() ) || ( itksys::SystemTools::LowerCase(extension) == ".dcm" ) )
    {
        std::string dicomDir = itksys::SystemTools::GetParentDirectory( fileName.c_str() );

        itk::GDCMSeriesFileNames::Pointer FileNameGenerator = itk::GDCMSeriesFileNames::New();
        FileNameGenerator->SetUseSeriesDetails( true );
        FileNameGenerator->SetDirectory( dicomDir );

        typedef const std::vector< std::string > ContainerType;
        const ContainerType & seriesUIDs = FileNameGenerator->GetSeriesUIDs();

        typedef typename itk::ImageSeriesReader< ImageType > ReaderType;
        typename ReaderType::Pointer reader = ReaderType::New();
        reader->SetFileNames( FileNameGenerator->GetFileNames( seriesUIDs[0] ) );
        reader->SetImageIO( dicomIO );
        try
        {
            reader->Update();
        }
        catch( itk::ExceptionObject & err )
        {
            std::cout << "Caught an exception: " << std::endl;
            std::cout << err << " " << __FILE__ << " " << __LINE__ << std::endl;
            throw;
        }
        catch( ... )
        {
            std::cout << "Error while reading in image for patient " << fileName << std::endl;
            throw;
        }
        image = reader->GetOutput();
        image->DisconnectPipeline();
        //image->ReleaseDataFlagOn();
    }
    else
    {
        typedef itk::ImageFileReader< ImageType > ReaderType;
        typename ReaderType::ConstPointer reader = ReaderType::New();
        reader->SetFileName( fileName.c_str() );
        try
        {
            reader->Update();
        }
        catch( itk::ExceptionObject & err )
        {
            std::cout << "Caught an exception: " << std::endl;
            std::cout << err << " " << __FILE__ << " " << __LINE__ << std::endl;
            throw;
        }
        catch( ... )
        {
            std::cout << "Error while reading in image" << fileName << std::endl;
            throw;
        }
        image = reader->GetOutput();
        image->DisconnectPipeline();
        //image->ReleaseDataFlagOn();
    }
    return image;
}





/**

*/
template< typename ImageType1, typename ImageType2 >
bool ImagePhysicalDimensionsAreIdentical( const ImageType1 *inputImage1, const ImageType2 *inputImage2 )
{
    bool same = true;
    same &= ( inputImage1->GetDirection() == inputImage2->GetDirection() );
    same &= ( inputImage1->GetSpacing() == inputImage2->GetSpacing() );
    same &= ( inputImage1->GetOrigin() == inputImage2->GetOrigin() );
    return same;
}

/**

*/
template< typename ImageType >
typename ImageType::Pointer OrientImage( const ImageType *inputImage, itk::SpatialOrientation::ValidCoordinateOrientationFlags orient)
{
    typedef itk::OrientImageFilter< ImageType, ImageType > OrientImageFilterType;
    typename OrientImageFilterType::Pointer orienter = OrientImageFilterType::New();
    orienter->SetDesiredCoordinateOrientation(orient);
    orienter->UseImageDirectionOn();
    orienter->SetInput(inputImage);
    orienter->Update();
    
    typename ImageType::Pointer image = orienter->GetOutput();
    image->DisconnectPipeline();
    //image->ReleaseDataFlagOn();
    return image;
}

/**

*/
template< typename ImageType >
typename ImageType::Pointer OrientImage( const ImageType *inputImage, const typename ImageType::DirectionType &dirCosines )
{
    return OrientImage< ImageType >( inputImage, SOAdapterType().FromDirectionCosines( dirCosines) );
}

/**

*/
template< typename ImageType >
typename ImageType::Pointer OrientImage1( const ImageType *inputImage, const typename ImageType::DirectionType &dirCosines )
{
    typename ImageType::ConstPointer constImg( inputImage );
    return OrientImage< ImageType >( constImg, SOAdapterType().FromDirectionCosines(dirCosines) );
}

/**

*/
template< typename ImageType >
typename ImageType::Pointer OrientImage2( const ImageType *inputImage, itk::SpatialOrientation::ValidCoordinateOrientationFlags orient )
{
    typename ImageType::ConstPointer constImg(inputImage);
    return OrientImage< ImageType >( constImg, orient );
}

/**

*/
template < typename ImageType >
typename ImageType::Pointer ReadImageAndOrient( const std::string &filename )
{
    typename ImageType::Pointer img = ReadImage< ImageType >( filename );
    typename ImageType::ConstPointer constImg( img );
    typename ImageType::Pointer image = OrientImage< ImageType >( constImg, itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIP );
    return image;
}

/**

*/
template < typename ImageType >
typename ImageType::Pointer ReadImageAndOrient( const std::string &filename, itk::SpatialOrientation::ValidCoordinateOrientationFlags orient )
{
    typename ImageType::Pointer img = ReadImage< ImageType >( filename );
    typename ImageType::ConstPointer constImg( img );
    typename ImageType::Pointer image = OrientImage< ImageType >( constImg, orient );
    return image;
}

/**

*/
template <typename ImageType>
typename ImageType::Pointer ReadImageAndOrient( const std::string &filename, const DirectionType &dir )
{
    return ReadImageAndOrient< ImageType >( filename, SOAdapterType().FromDirectionCosines( dir ) );
}

/**

*/
template< typename ImageType >
typename ImageType::Pointer ReadImageCoronal( const std::string &fileName )
{
    DirectionType CORdir = SOAdapterType().ToDirectionCosines( itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIP );
    return ReadImageAndOrient< ImageType >( fileName, CORdir );
}

/**

*/
template< typename ImageType >
void WriteImage( const ImageType *image, const std::string &filename )
{
    typedef itk::ImageFileWriter< ImageType > WriterType;
    typename  WriterType::Pointer writer = WriterType::New();
    writer->UseCompressionOn();
    writer->SetFileName( filename.c_str() );
    writer->SetInput(image);
    try
    {
        writer->Update();
    }
    catch( itk::ExceptionObject & err )
    {
        std::cout << "Exception Object caught: " << std::endl;
        std::cout << err << std::endl;
        throw;
    }
}

/**

*/
template< typename InputImageType, typename OutputImageType >
typename OutputImageType::Pointer TypeCast( const InputImageType *input )
{
	typedef itk::CastImageFilter< InputImageType, OutputImageType > CastToOutputImageType;
	typename CastToOutputImageType::Pointer toNewType = CastToOutputImageType::New();
	toNewType->SetInput( input );

	try {
		toNewType->Update();
	}
	catch(... ) {
		std::cout << "Error while casting an image" << std::endl;
		throw;
	}

    typename OutputImageType::Pointer image = toNewType->GetOutput();
    image->DisconnectPipeline();
//    image->ReleaseDataFlagOn();
	return image;
}

/**
  * Convert images from one type to another with explicit min and max values.
  * NOTE: The original range of the image is determined explicitly from the data,
  * and then linearly scaled into the range specified.
*/
template< typename InputImageType, typename OutputImageType >
typename OutputImageType::Pointer ScaleAndCast( const InputImageType *image,
                                                const typename OutputImageType::PixelType OutputMin,
                                                const typename OutputImageType::PixelType OutputMax )
{
    typedef itk::RescaleIntensityImageFilter<InputImageType, OutputImageType> R2CRescaleFilterType;
    typename R2CRescaleFilterType::Pointer RealToProbMapCast = R2CRescaleFilterType::New();
    RealToProbMapCast->SetOutputMinimum( OutputMin );
    RealToProbMapCast->SetOutputMaximum( OutputMax );
    RealToProbMapCast->SetInput(image);
    try
    {
        RealToProbMapCast->Update();
    }
    catch( itk::ExceptionObject & e )
    {
        throw;
    }
    typename OutputImageType::Pointer returnScaled = RealToProbMapCast->GetOutput();
    return returnScaled;
}

/**
  * This function will do a type cast if the OutputImageType
  * intensity range is larger than the input image type range.
  * If the OutputImageType range is smaller, then a Scaling will occur.
*/
template< typename InputImageType, typename OutputImageType >
typename OutputImageType::Pointer PreserveCast( const InputImageType *image )
{
    const typename InputImageType::PixelType inputmin = itk::NumericTraits<typename InputImageType::PixelType>::min();
    const typename InputImageType::PixelType inputmax = itk::NumericTraits<typename InputImageType::PixelType>::max();
    const typename OutputImageType::PixelType outputmin = itk::NumericTraits<typename OutputImageType::PixelType>::min();
    const typename OutputImageType::PixelType outputmax = itk::NumericTraits<typename OutputImageType::PixelType>::max();
    if( ( inputmin >= outputmin ) && ( inputmax <= outputmax ) )
    {
        return TypeCast< InputImageType, OutputImageType >( image );
    }
    else
    {
        return ScaleAndCast< InputImageType, OutputImageType >( image, outputmin, outputmax );
    }
}

/**

*/
template< typename ImageType >
typename ImageType::Pointer CopyImage( const ImageType *input )
{
    typedef itk::ImageDuplicator< ImageType > ImageDuplicatorType;
    typename ImageDuplicatorType::Pointer MyDuplicator = ImageDuplicatorType::New();
    MyDuplicator->SetInputImage( input );
    MyDuplicator->Update();
    return MyDuplicator->GetModifiableOutput();
}

/**
  * Common code for allocating an image, allowing the
  * region and spacing to be explicitly set.
*/
template< typename TemplateImageType, typename OutputImageType >
typename OutputImageType::Pointer AllocateImageFromRegionAndSpacing( const typename TemplateImageType::RegionType &region,
                                                                     const typename TemplateImageType::SpacingType &spacing )
{
    typename OutputImageType::Pointer image;
    image = OutputImageType::New();
    image->SetSpacing( spacing );
    //    image->SetLargestPossibleRegion(region);
    //    image->SetBufferedRegion(region);
    image->SetRegions(region);
    image->Allocate();
    return image;
}

/**

*/
template< typename ImageType >
typename ImageType::Pointer AllocateImageFromRegionAndSpacing( const typename ImageType::RegionType &region,
                                                               const typename ImageType::SpacingType &spacing )
{
    return AllocateImageFromRegionAndSpacing< ImageType, ImageType >( region, spacing );
}

/**
  * AllocateImageFromExample creates and allocates an image of the type OutputImageType,
  * using TemplateImageType as the source of size and spacing...
*/
template< typename TemplateImageType, typename OutputImageType >
typename OutputImageType::Pointer AllocateImageFromExample( const typename TemplateImageType::Pointer &TemplateImage )
{
    typename OutputImageType::Pointer rval = OutputImageType::New();
    rval->CopyInformation( TemplateImage );
    rval->SetRegions( TemplateImage->GetLargestPossibleRegion() );
    rval->Allocate();
    return rval;
}

/**
  * Convenience function where template and output images type are the same
*/
template< typename ImageType >
typename ImageType::Pointer AllocateImageFromExample( const ImageType *TemplateImage )
{
    return AllocateImageFromExample< ImageType, ImageType >( TemplateImage );
}

/**

*/
template< typename TInputImageType, typename TOutputImageType >
typename TOutputImageType::Pointer ResampleImage( const TInputImageType *inputImage,
		                                          typename TInputImageType::SpacingType newSpace,
		                                          typename TInputImageType::SizeType newSize,
		                                          typename TInputImageType::PointType newOrigin,
		                                          typename TInputImageType::IndexType newStartIndex,
		                                          typename TInputImageType::DirectionType newDirection,
		                                          int type = LINEAR_INTERPOLATION )
{
    /// Interpolator types
    typedef itk::LinearInterpolateImageFunction< TInputImageType, double > LinearInterpolatorType;
    typedef itk::BSplineInterpolateImageFunction< TInputImageType, double > BSplineInterpolatorType;
    typedef itk::NearestNeighborInterpolateImageFunction< TInputImageType, double > NNInterpolatorType;

  	/// Transformation
  	typedef itk::IdentityTransform< double, TInputImageType::ImageDimension > TransformType;
  	typename TransformType::Pointer transform = TransformType::New();
  	transform->SetIdentity();

    /// Resampling filter
  	typedef itk::ResampleImageFilter< TInputImageType, TOutputImageType > ResampleFilterType;
  	typename ResampleFilterType::Pointer resampler = ResampleFilterType::New();
  	resampler->SetTransform( transform );

    typename ResampleFilterType::InterpolatorPointerType interpolator;
    switch( type )
    {
      case LINEAR_INTERPOLATION:
      default:
    	    interpolator = LinearInterpolatorType::New();
    	    break;
      case BSPLINE_INTERPOLATION:
    	    interpolator = BSplineInterpolatorType::New();
    	    break;
      case NEAREST_NEIGHBOR_INTERPOLATION:
            interpolator = NNInterpolatorType::New();
    }

  	resampler->SetInput( inputImage );
  	resampler->SetDefaultPixelValue( 0 );
    resampler->SetInterpolator( interpolator );
  	resampler->SetOutputSpacing( newSpace );
  	resampler->SetOutputOrigin( newOrigin );
  	resampler->SetOutputStartIndex( newStartIndex );
  	resampler->SetOutputDirection( newDirection );
  	resampler->SetSize( newSize );

    try {
		  resampler->Update();
    }
    catch(... ) {
      std::cout << "Error while resampling an image" << std::endl;
      throw;
    }

    typename TOutputImageType::Pointer img = resampler->GetOutput();
    img->DisconnectPipeline();
  	return img;
}

/**
  Get the PixelType and ComponentType from fileName
*/
void GetImageType( std::string fileName,
                   itk::ImageIOBase::IOPixelType &pixelType,
                   itk::ImageIOBase::IOComponentType &componentType )
{
    typedef itk::Image< unsigned char, 3 > ImageType;

    typedef itk::ImageFileReader< ImageType > ImageFileReaderType;
    ImageFileReaderType::Pointer imageReader = ImageFileReaderType::New();
    imageReader->SetFileName( fileName.c_str() );
    imageReader->UpdateOutputInformation();

    pixelType = imageReader->GetImageIO()->GetPixelType();
    componentType = imageReader->GetImageIO()->GetComponentType();
}

/**
  Get the PixelTypes and ComponentTypes from fileNames
*/
void GetImageTypes ( std::vector<std::string> fileNames,
                     std::vector<itk::ImageIOBase::IOPixelType> &pixelTypes,
                     std::vector<itk::ImageIOBase::IOComponentType> &componentTypes )
{
    pixelTypes.clear();
    componentTypes.clear();

    // For each file, find the pixel and component type
    for (std::vector<std::string>::size_type i = 0; i < fileNames.size(); i++)
    {
        itk::ImageIOBase::IOPixelType pixelType;
        itk::ImageIOBase::IOComponentType componentType;

        GetImageType( fileNames[i], pixelType, componentType );
        pixelTypes.push_back(pixelType);
        componentTypes.push_back(componentType);
    }
}

/**

*/
template< class T >
void AlignVolumeCenters( T *fixed, T *moving, typename T::PointType &origin )
{
    // compute the center of fixed
    typename T::PointType fixedCenter;
    {
        itk::ContinuousIndex< double, T::ImageDimension > centerIndex;
        typename T::SizeType size = fixed->GetLargestPossibleRegion().GetSize();
        for (unsigned int i = 0; i < T::ImageDimension; i++)
        {
            centerIndex[i] = static_cast< double >( (size[i]-1)/2.0 );
        }
        fixed->TransformContinuousIndexToPhysicalPoint( centerIndex, fixedCenter );
    }

    // compute the center of moving
    typename T::PointType movingCenter;
    {
        itk::ContinuousIndex< double, T::ImageDimension > centerIndex;
        typename T::SizeType size = moving->GetLargestPossibleRegion().GetSize();
        for (unsigned i = 0; i < T::ImageDimension; i++)
        {
            centerIndex[i] = static_cast <double >( (size[i]-1)/2.0 );
        }
        moving->TransformContinuousIndexToPhysicalPoint( centerIndex, movingCenter );
    }

    for (unsigned int j = 0; j < fixedCenter.Size(); j++)
    {
        origin[j] = moving->GetOrigin()[j] - (movingCenter[j] - fixedCenter[j]);
    }
}

/**

*/
void GetImageInformation( std::string fileName,
                          itk::ImageIOBase::IOComponentType &componentType,
                          unsigned int & dimension )
{
    // Find out the component type of the image in file
    typedef itk::ImageIOBase::IOComponentType PixelType;

    itk::ImageIOBase::Pointer imageIO =
    itk::ImageIOFactory::CreateImageIO( fileName.c_str(),
                                        itk::ImageIOFactory::ReadMode );
    if( !imageIO )
    {
        std::cerr << "NO IMAGEIO WAS FOUND" << std::endl;
        return;
    }

    // Now that we found the appropriate ImageIO class, ask it to
    // read the meta data from the image file.
    imageIO->SetFileName( fileName.c_str() );
    imageIO->ReadImageInformation();

    componentType = imageIO->GetComponentType();
    dimension = imageIO->GetNumberOfDimensions();
}

/**

*/
template<class TValue>
TValue Convert( std::string optionString )
{
    TValue value;
    std::istringstream iss( optionString );

    iss >> value;
    return value;
}

/**

*/
template<class TValue>
std::vector<TValue> ConvertVector( std::string optionString )
{
    std::vector<TValue>    values;
    std::string::size_type crosspos = optionString.find( 'x', 0 );

    if ( crosspos == std::string::npos )
    {
        values.push_back( Convert<TValue>( optionString ) );
    }
    else
    {
        std::string element = optionString.substr( 0, crosspos );
        TValue      value;
        std::istringstream iss( element );
        iss >> value;
        values.push_back( value );
        while ( crosspos != std::string::npos )
        {
            std::string::size_type crossposfrom = crosspos;
            crosspos = optionString.find( 'x', crossposfrom + 1 );
            if ( crosspos == std::string::npos )
            {
                element = optionString.substr( crossposfrom + 1, optionString.length() );
            }
            else
            {
                element = optionString.substr( crossposfrom + 1, crosspos );
            }
            std::istringstream iss2( element );
            iss2 >> value;
            values.push_back( value );
        }
    }
    return values;
}


} // End namespace utils

} // End namespace bip

#endif
