
// TODO: Add OpenCL kernel code here.
#define binSize 256

/* This cl introduce one global imageData,
 * and compute different result in each workItem,
 * then combine workItem in the same workgroup
 */

__kernel
void histogram(__global const uchar* imageData, 
				int width,
				int height,
				__local float* workItem, 
				__global float* workGroup)
{
	//Returns the unique local work-item ID.
    size_t localId = get_local_id(0);
	//Returns the unique global work-item ID
    size_t globalId = get_global_id(0);
	//Returns the work-group ID.
    size_t groupId = get_group_id(0);
	//Returns the number of local work-items.
    size_t groupSize = get_local_size(0);

    //initialize the work item
    for(int i = 0; i < binSize; ++i)
        workItem[localId * binSize + i] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    //compute the local histogram in a work item
    for(int i = 0; i < binSize; ++i)
    {
        uint value = imageData[globalId * binSize + i];
        workItem[localId * binSize + value] += (float)256/(width * height);
    } 
    barrier(CLK_LOCAL_MEM_FENCE); 
    
   //combine the result in work item of a same workgroup
	float temp = 0;
    for(int i = 0; i < binSize / groupSize; i++){   
        for(int j = 0; j < groupSize; j++)
            temp += (float)workItem[j * binSize + i * groupSize + localId];            
        workGroup[groupId * binSize + i * groupSize + localId] = temp;
    }
}