/* LINITIM.C - Initialize a diffraction image.

   Author: Mike Wall
   Date: 9/27/95
   Version: 1.

*/

#include<mwmask.h>

DIFFIMAGE *linitim(void)
{
  size_t
    i,
    num_read;
  
  DIFFIMAGE
    *imdiff;
  /*
   * Allocate memory:
   */
  
  imdiff = (DIFFIMAGE *)malloc(sizeof(DIFFIMAGE));
  if ((!imdiff)) {
    printf("\nLINITIM:  Unable to allocate all memory (imdiff).\n");
    imdiff = NULL;
    goto CloseShop;
  }
  
  
  imdiff->vpixels = DEFAULT_VSIZE;
  imdiff->hpixels = DEFAULT_HSIZE;
  imdiff->image_length = DEFAULT_IMAGELENGTH;
  imdiff->punch_tag = PUNCH_TAG;
  imdiff->overload_tag = DEFAULT_OVERLOAD_TAG;
  imdiff->ignore_tag = DEFAULT_IGNORE_TAG;
  imdiff->lattice_ignore_tag = DEFAULT_LATTICE_IGNORE_TAG;
  imdiff->header_length = DEFAULT_HEADER_LENGTH;
  imdiff->mask_inner_radius = DEFAULT_INNER_RADIUS;
  imdiff->mask_outer_radius = DEFAULT_OUTER_RADIUS;
  imdiff->origin.r = DEFAULT_IMAGE_ORIGIN;
  imdiff->origin.c = DEFAULT_IMAGE_ORIGIN;
  imdiff->distance_mm = DEFAULT_DISTANCE_MM;
  imdiff->beam_mm.x = DEFAULT_X_BEAM;
  imdiff->beam_mm.y = DEFAULT_Y_BEAM;
  imdiff->pixel_size_mm = DEFAULT_PIXEL_SIZE_MM;
  imdiff->spindle_deg = DEFAULT_SPINDLE_DEG;
  imdiff->rfile_mask_tag = DEFAULT_RFILE_MASK_TAG;
  imdiff->weights_height = DEFAULT_WEIGHTS_DIMENSION;
  imdiff->weights_width = DEFAULT_WEIGHTS_DIMENSION;
  imdiff->mode_height = DEFAULT_MODE_DIMENSION;
  imdiff->mode_width = DEFAULT_MODE_DIMENSION;
  imdiff->window_upper.r = DEFAULT_WINDOW_UPPER;
  imdiff->window_upper.c = DEFAULT_WINDOW_UPPER;
  imdiff->window_lower.r = DEFAULT_WINDOW_LOWER;
  imdiff->window_lower.c = DEFAULT_WINDOW_LOWER;
  imdiff->polarization = DEFAULT_POLARIZATION;
  imdiff->polarization_offset = DEFAULT_POLARIZATION_OFFSET;
  imdiff->feature_count = 0;
  imdiff->cell.a = DEFAULT_CELL_A;
  imdiff->cell.b = DEFAULT_CELL_B;
  imdiff->cell.c = DEFAULT_CELL_C;
  imdiff->cell.alpha = DEFAULT_CELL_ALPHA;
  imdiff->cell.beta = DEFAULT_CELL_BETA;
  imdiff->cell.gamma = DEFAULT_CELL_GAMMA;
  imdiff->wavelength = DEFAULT_WAVELENGTH;
  imdiff->cassette.x = DEFAULT_CASSETTE_ROTX;
  imdiff->cassette.y = DEFAULT_CASSETTE_ROTY;
  imdiff->cassette.z = DEFAULT_CASSETTE_ROTZ;
  imdiff->amplitude = DEFAULT_AMPLITUDE;
  imdiff->pitch = DEFAULT_PITCH;

  /*
   * Allocate memory for arrays:
   */
  
  imdiff->rfile = (RFILE_DATA_TYPE *)calloc(MAX_RFILE_LENGTH,
					    sizeof(RFILE_DATA_TYPE));
  imdiff->imscaler = (RFILE_DATA_TYPE *)calloc(MAX_RFILE_LENGTH,
					       sizeof(RFILE_DATA_TYPE));
  imdiff->imoffsetr = (RFILE_DATA_TYPE *)calloc(MAX_RFILE_LENGTH,
						sizeof(RFILE_DATA_TYPE));
  imdiff->mask  = (struct rccoords *)malloc(sizeof(struct rccoords)*
					    MAX_MASK_PIXELS);
  imdiff->image = (IMAGE_DATA_TYPE *)calloc(imdiff->image_length,
					    sizeof(IMAGE_DATA_TYPE));
  imdiff->header = (char *)malloc(sizeof(char)*imdiff->header_length);
  imdiff->overload = (struct rccoords *)malloc(sizeof(struct rccoords)*
					       MAX_OVERLOADS);
  imdiff->peak = (struct xycoords *)malloc(sizeof(struct xycoords)*MAX_PEAKS);
  imdiff->weights = (WEIGHTS_DATA_TYPE *)malloc(sizeof(WEIGHTS_DATA_TYPE)*
						(MAX_WEIGHTS_DIMENSION*
						 MAX_WEIGHTS_DIMENSION + 1));
  imdiff->rdata = (RDATA_DATA_TYPE *)
    malloc(sizeof(RDATA_DATA_TYPE) * MAX_RFILE_LENGTH);
  imdiff->feature = (DIFFUSE_FEATURE *) malloc(sizeof(DIFFUSE_FEATURE) * 
					       MAX_DIFFUSE_FEATURES);
  if (!imdiff->rfile || !imdiff->imscaler || !imdiff->imoffsetr || 
      !imdiff->mask || !imdiff->image || !imdiff->header || 
      !imdiff->overload || !imdiff->peak || !imdiff->weights ||
      !imdiff->rdata || !imdiff->feature) {
    printf("\nLINITIM:  Unable to allocate all memory.\n");
    imdiff = NULL;
    goto CloseShop;
  }
  
  for(i=0; i<MAX_WEIGHTS_DIMENSION*MAX_WEIGHTS_DIMENSION; i++) {
    imdiff->weights[i] = 1.;
  }
  for(i=0; i<MAX_RFILE_LENGTH; i++) {
    imdiff->rdata[i].count = 0;
    imdiff->rdata[i].allocate_flag = !VALUE_ALLOCATED;
  }
  
 CloseShop:

  return(imdiff);
}
