#ifndef __bipMiscellaneous__
#define __bipMiscellaneous__

// ITK includes
#include <itkArray.h>
#include <itkFixedArray.h>
#include <itkImage.h>
#include <itkImageToListSampleAdaptor.h>
#include <itkIndex.h>
#include <itkListSample.h>
#include <itkPoint.h>
#include <itkSubsample.h>
#include <itkVariableLengthVector.h>
#include <itkVector.h>
#include <itkVectorImage.h>
#include <itkVectorIndexSelectionCastImageFilter.h>

// STL includes
#include <algorithm>
#include <map>
#include <vector>
#include <set>

// STDIO includes
#include <math.h>


namespace bip
{
	namespace misc
	{
		// Bedrock types
		typedef double 		  InternalPixelType;
		typedef float 		  AtlasPixelType;
		typedef unsigned char MaskPixelType;
		typedef unsigned char OutputPixelType;
		typedef unsigned char LabeledPixelType;
		const int Dimension = 3;


		// Image types
		typedef itk::Image< InternalPixelType, Dimension > 		 InputImageType;
		typedef itk::Image< MaskPixelType, Dimension > 			 MaskImageType;
		typedef itk::Image< OutputPixelType, Dimension > 		 OutputImageType;
		typedef itk::Image< LabeledPixelType, Dimension > 		 LabeledImageType;
		typedef itk::VariableLengthVector< InternalPixelType > 	 VariableLengthVectorType; // sample
		typedef itk::VectorImage< AtlasPixelType, Dimension > 	 AtlasVectorImageType;
		typedef itk::VectorImage< InternalPixelType, Dimension > VectorImageType;
		
		
		// Statistics
		typedef itk::Statistics::ListSample< VariableLengthVectorType >      ListSampleType;
		typedef itk::Statistics::Subsample< ListSampleType > 				 SubSampleType;
		typedef itk::Statistics::ImageToListSampleAdaptor< VectorImageType > ListSampleAdaptorType;


		// Index and point
		typedef itk::Index< 3 > IndexType;
		typedef std::vector< IndexType > ListIndexType;
		typedef itk::VectorIndexSelectionCastImageFilter< VectorImageType, InputImageType > IndexSelectionType;
		typedef itk::Point< double, Dimension > PointType;


		// Neighbor type
		struct NeighborType
		{
			double dist_to_center;

			std::vector< double > neighbor_g;
			std::vector< double > neighbor_prior;
			std::vector< double > neighbor_posterior;
			std::vector< double > neighbor_u; // Gamma weight

			VariableLengthVectorType neighbor_sample;
			InputImageType::IndexType neighbor_index;
		};


		// We need a custom compare in order to std::map correctly order (and find) indexes
		// References: https://stackoverflow.com/questions/2620862/using-custom-stdset-comparator
		//             https://stackoverflow.com/questions/5733254/create-an-own-comparator-for-map
		struct myMapComp
		{
			bool operator ()(const InputImageType::IndexType &idx1, const InputImageType::IndexType &idx2) const
			{
				if(idx1[0] != idx2[0])
					return idx1[0] < idx2[0];
				else if(idx1[1] != idx2[1])
					return idx1[1] < idx2[1];
				else
					return idx1[2] < idx2[2];
			}
		};


		// Sample type
		struct SampleType
		{
			std::map< InputImageType::IndexType, int, myMapComp > neighbors_idx; // maps a neighbor index to its corresponding vector index
			std::vector< double > alpha;
			std::vector< double > g;
			std::vector< double > prior;
			std::vector< double > posterior;
			std::vector< double > sample_u; // Gamma weight
			std::vector< NeighborType > neighbors;

			std::vector< IndexType > neighborhood_idxs;
			
			InputImageType::IndexType sample_index;
			VariableLengthVectorType sample;
		};


		// Simple function to calculate the
		// Euclidean distance between two points
		double GetDistance(InputImageType::IndexType idx1, InputImageType::IndexType idx2, int dim)
		{
			double dist = 0.0;
			for(int i = 0; i < dim; ++i)
				dist += ((idx1[i] - idx2[i]) * (idx1[i] - idx2[i]));

			dist = sqrt(dist);

			return 1.0 / (1.0 + dist * dist);
		}

		template< typename InputImageType, typename OutputImageType >
		typename OutputImageType::Pointer allocate_image(typename InputImageType::Pointer img)
		{
		    typename OutputImageType::Pointer alloc_img = OutputImageType::New();
		    typename OutputImageType::RegionType region = img->GetLargestPossibleRegion();
		    alloc_img->SetRegions(region);
		    alloc_img->SetOrigin(img->GetOrigin());
		    alloc_img->SetSpacing(img->GetSpacing());
		    alloc_img->SetDirection(img->GetDirection());
		    alloc_img->Allocate();

		    itk::ImageRegionIterator< OutputImageType > iterator(alloc_img, alloc_img->GetLargestPossibleRegion());
		    for(iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
		        iterator.Set(0);

		    return alloc_img;
		}

		template< typename InputImageType, typename OutputImageType >
		typename OutputImageType::Pointer allocate_mask_image(typename InputImageType::Pointer img)
		{
		    typename OutputImageType::Pointer alloc_img = OutputImageType::New();
		    typename OutputImageType::RegionType region = img->GetLargestPossibleRegion();
		    alloc_img->SetRegions(region);
		    alloc_img->SetOrigin(img->GetOrigin());
		    alloc_img->SetSpacing(img->GetSpacing());
		    alloc_img->SetDirection(img->GetDirection());
		    alloc_img->Allocate();

		    itk::ImageRegionIterator< OutputImageType > iterator(alloc_img, alloc_img->GetLargestPossibleRegion());
		    for(iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
		        iterator.Set(1);

		    return alloc_img;
		}

	}
}

#endif