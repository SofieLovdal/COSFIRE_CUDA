/*
* CUDA kernel for 2D matrix shift. Ignores borders. 
* Sofie Lovdal 18.6.2018
*/

__global__ void shiftPixels(double * output, double * const input, 
						  unsigned int const numRows, unsigned int const numCols,
						  double const rho, double const phi)
{
   
   /*global thread ID in x, y dimension*/
   const int colIdx = blockIdx.x*blockDim.x + threadIdx.x;
   const int rowIdx = blockIdx.y*blockDim.y + threadIdx.y;
    
   /*make sure we are within image*/
   if(colIdx>=numCols || rowIdx >= numRows) return;
   
   /*Consider one pixel at the time in the input image. Calculate, based on rho
    * and phi, how much this pixel should be shifted, and insert it in the
    * corresponding position in the output buffer. If pixel goes outside of
    * image, just exit. Here it would be good if the output buffer is from 
    * the beginning initialized to zero*/ 
   
   /*Linear index of the pixel to be considered*/
   int linearIdx = rowIdx*numCols + colIdx;
   double pixelValue = input[linearIdx];
   
   int deltax = (int)(-rho*cos(phi)); //FIX: ROUND TOWARDS 0. cos(pi/2)=0 so move 0 steps in x direction
   int deltay = (int)(-rho*sin(phi)); //sin(pi/2)=1 so move -rho steps in y direction (two pixels upwards, -2)
   
   if(colIdx+deltax<0 || colIdx+deltax>=numCols || rowIdx-deltay<0 || rowIdx-deltay >= numRows) return;
   
   /*y axis is opposite direction in image compared to cartesian coordinate system?*/
   int outputPixel = linearIdx + deltax - deltay*numCols; //or minus?
   
   output[outputPixel] = pixelValue;

}   
