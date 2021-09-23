/*
This file defines the abstract base class for storage classes, i.e., points on a manifold,
tangent vectors, vectors in ambient space, linear operator on a tangent space.
It uses copy-on-write strategy.

SmartSpace

---- WH
*/

#ifndef SMARTSPACE_H
#define SMARTSPACE_H

#include "Others/randgen.h"
#include <cstdarg>
#include <map>
#include "Others/def.h"

/*Define the namespace*/
namespace ROPTLIB{

#ifdef CHECKMEMORYDELETED
	extern std::map<integer *, integer> *CheckMemoryDeleted;
#endif

	class SmartSpace{
	public:
		/*Initialize the SmartSpace. If one want to create a 10 by 3 by 2 by 5 tensor,
		then call this function by
		Initialization(4, 10, 3, 2, 5);
		The first argument indicates the number of following parameters.
		The next 4 arguments means the dimensions of the tensor.
		This SmartSpace will not allocate memory in this function. */
		virtual void Initialization(integer numberofdimensions, ...);

		/*Copy this SmartSpace to "eta" SmartSpace. After calling this function,
		this SmartSpace and "eta" SmartSpace will use same space to store data. */
		virtual void CopyTo(SmartSpace &eta) const;

		/*Randomly create this SmartSpace. In other words, the space will be allocated based
		on the size. Then each entry in the space will be generated by the uniform distribution in [start, end].*/
		virtual void RandUnform(realdp start = 0, realdp end = 1);

		/*Randomly create this SmartSpace. In other words, the space will be allocated based
		on the size. Then each entry in the space will be generated by the normal distribution with mean and variance.*/
		virtual void RandGaussian(realdp mean = 0, realdp variance = 1);
        
        /*Create this SmartSpace. All entries are zero*/
        virtual void SetToZeros(void);

		/*Obtain this SmartSpace's pointer which points to the data;
		Users are encouraged to call this function if they only need to use the data, not to modify the data.
		The data may be shared with other SmartSpace. Therefore, it is risky to modify the data.*/
		virtual const realdp *ObtainReadData(void) const;

		/*Obtain this SmartSpace's pointer which points to the data;
		Users are encourage to call this function if they want to overwrite the data without caring about its original data.
		If the data is shared with other SmartSpace, then new memory are allocated without copying the data to the new memory.*/
		virtual realdp *ObtainWriteEntireData(void);

		/*Obtain this SmartSpace's pointer which points to the data;
		If the data is shared with other SmartSpace, then new memory are allocated and the data are copied to the new memory.*/
		virtual realdp *ObtainWritePartialData(void);

		/*If the data is shared with other SmartSpace, then new memory are allocated without copying the data to the new memory.*/
		virtual void NewMemoryOnWrite(void);

		/*If the data is shared with other SmartSpace, then new memory are allocated and the data are copied to the new memory.*/
		virtual void CopyOnWrite(void);

		/*Return the number of SmartSpaces which points to the data this SmartSpace points to*/
		inline integer *GetSharedTimes(void) const { return sharedtimes; };

		/*Return the pointer which points to the data*/
		inline const realdp *GetSpace(void) const { return Space; };

		/*destruct this SpartSpace. If the data is only used by this SmartSpace, then delete the memory of the data.
		Otherwise, reduce the "*sharetimes" by 1.*/
		virtual ~SmartSpace(void) = 0;

		/*Print the data. The string "name" is to mark the output such that user can find the output easily.*/
		virtual void Print(const char *name = "") const;

		/*Get the array "size", which indicates the size of the data.*/
		inline const integer *Getsize(void) const { return size; };
        
		/*Get the length of the "size".*/
		inline integer Getls(void) const { return ls; };

		/*Get the total length of the data*/
		inline integer Getlength(void) const { return length; };

		/*This is used for creating a collection of multiple elements. it is not encouraged to be used by users. It initializes this SmartSpace by setting the parameters*/
		virtual void SetByParams(integer *size, integer ls, integer length, integer *sharedtimes, realdp *Space);
        
        /*This is used for creating a collection of multiple elements. it is not encouraged to be used by users. It initializes this SmartSpace by setting the parameters*/
        virtual void SetByParams(integer *insharedtimes, realdp *Space);
        
        /*This is not encouraged to be used by users. It deleted this SmartSpace by setting all parameters to be NULL*/
        virtual void DeleteBySettingNull(void);
        
		/*Delete current space*/
		virtual void Delete(void);
	protected:
		integer *size; /* size of the data*/
		integer ls; /*length of the array size*/
		integer length; /*the length of the data*/

		integer *sharedtimes; /*The number of use of the data*/
		realdp *Space; /*The pointer which points to the data*/

		/*allocate memory*/
		void NewMemory(void);
	};
}; /*end of ROPTLIB namespace*/

#endif