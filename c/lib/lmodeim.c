/* LMODEIM.C - Remove the peaks from an image by mode filetering using
   a specified mask size.
   
   Author: Mike Wall 
   Date: 2/2/95
   Version: 1.
   
   "modeim <image in> <image out> <mask size> <bin size>"

   Input is ascii coordinates file.  Output is 16-bit 
   image of specified size (1024 x 1024 default).

   Reentry codes (imdiff->reentry):
   0 = Enter fresh without intent for future reentry (initialize and free)
   1 = Enter fresh with intent for future reentry (initialize without free)
   2 = Reentry with intent for future reentry (no initialize, no free)
   3 = Reentry without intent for future reentry (no initialize, but do free)
*/

#include<mwmask.h>
#ifdef USE_OPENMP
#include<omp.h>
#endif
#include<time.h>

#ifdef USE_OFFLOAD
#pragma omp declare target
#endif

static size_t x=123456789, y=362436069, z=521288629;

// Code taken from xorshf96()

size_t rand_local(void) {          //period 2^96-1
size_t t;
    x ^= x << 16;
    x ^= x >> 5;
    x ^= x << 1;

   t = x;
   x = y;
   y = z;
   z = t ^ x ^ y;

  return z;
}

double log_local(double x) {
  int n,N=10,onefac=1;
  double y,xfac,f = 0.0;

  xfac = y = x-1.;

  for (n=1;n<=10;n++) {
    f += (double)onefac*xfac/(double)n;
    onefac *= -1;
    xfac *= y;
  }
  return(f);
}

void swap(size_t *a, size_t *b) 
{ 
  size_t t = *a; 
  *a = *b; 
  *b = t; 
} 
  
int partition(size_t arr[], int l, int h) 
{ 
  size_t x = arr[h]; 
  int i = (l - 1); 
  int j;

  for (j = l; j <= h - 1; j++) { 
    if (arr[j] <= x) { 
      i++; 
      swap(&arr[i], &arr[j]); 
    } 
  } 
  swap(&arr[i + 1], &arr[h]); 
  return (i + 1); 
} 

void quickSortIterative(size_t arr[], size_t stack[], int l, int h) 
{ 
  // initialize top of stack 
  int top = -1; 
  
  // push initial values of l and h to stack 
  stack[++top] = l; 
  stack[++top] = h; 
  
  // Keep popping from stack while is not empty 
  while (top >= 0) { 
    // Pop h and l 
    h = stack[top--]; 
    l = stack[top--]; 
  
    // Set pivot element at its correct position 
    // in sorted array 
    int p;
    p = partition(arr, l, h); 
  
    // If there are elements on left side of pivot, 
    // then push left side to stack 
    if (p - 1 > l) { 
      stack[++top] = l; 
      stack[++top] = p - 1; 
    } 
  
    // If there are elements on right side of pivot, 
    // then push right side to stack 
    if (p + 1 < h) { 
      stack[++top] = p + 1; 
      stack[++top] = h; 
    } 
  } 
}

void quickSortList(size_t arr[], size_t stack[], size_t num_arrays, size_t array_size) {
  size_t i;
#ifdef DEBUG
  printf("quickSortList: num_arrays, array_size = %ld, %ld\n",num_arrays,array_size);
#endif
#ifdef USE_OPENMP
#ifdef USE_OFFLOAD
#pragma omp target map(to:num_arrays,array_size)
#pragma omp teams distribute parallel for schedule(static,1)
#else
#pragma omp parallel for shared(stack,arr) 
#endif
#endif
  for (i = 0; i < num_arrays; i++) {
    size_t *this_window = &arr[i*array_size];
    size_t *this_stack = &stack[i*array_size];
    quickSortIterative(this_window,this_stack,0,array_size-1);
  }
}

void insertion_sort(size_t *a,int first,int last) {
  int len;
  int i=1+first;
  len = last - first + 1;
  while (i < len) {
    size_t x = a[i];
    int j = i - 1;
    while (j >= 0 && a[j] > x) {
      a[j + 1] = a[j];
      j = j - 1;
    }
    a[j+1] = x;
    i = i + 1;
  }
}

void quicksort(size_t *a,int first,int last){
  int i, j, pivot;
  size_t temp;

  if(first<last){
    pivot=first;
    i=first;
    j=last;

    while(i<j){
      while(a[i]<=a[pivot]&&i<last)
	i++;
      while(a[j]>a[pivot])
	j--;
      if(i<j){
	temp=a[i];
	a[i]=a[j];
	a[j]=temp;
      }
    }

    temp=a[pivot];
    a[pivot]=a[j];
    a[j]=temp;
    quicksort(a,first,j-1);
    quicksort(a,j+1,last);

  }
}
#ifdef USE_OFFLOAD
#pragma omp end declare target
#endif

int lmodeim(DIFFIMAGE *imdiff_in)
{
  RCCOORDS_DATA 
    half_height,
    half_width,
    n,
    m; 

  static IMAGE_DATA_TYPE
    //IMAGE_DATA_TYPE
    *image = NULL;

  IMAGE_DATA_TYPE
    maxval,
    minval,
    binsize;
  
  int
    return_value = 0;

  static size_t
    //size_t
    *image_mode = NULL,
    *window = NULL,
    *nvals = NULL,
    *stack = NULL;

  DIFFIMAGE *imdiff;

  int reentry = imdiff_in->reentry;

  size_t wlen = (imdiff_in->mode_height+2)*(imdiff_in->mode_width+2);
    
  int pidx;

  double tic, toc, tmkarr, tsort, tmkimg;

  size_t
    i,
    j;

  static size_t 
    //size_t 
    image_length = 0,
    hpixels = 0,
    vpixels = 0;

  size_t jblock, iblock;

  size_t
    num_jblocks = LUNUS_NUM_JBLOCKS,
    num_iblocks = LUNUS_NUM_IBLOCKS;

  static size_t 
    num_per_jblock, 
    num_per_iblock;

  //  printf("omp_get_num_teams() = %d, omp_get_num_threads() = %d\n",omp_get_num_teams(), omp_get_num_threads());

  //  fflush(stdout);

  imdiff = &imdiff_in[0];

  if (reentry == 2 || reentry == 3) {
    if (hpixels != imdiff->hpixels || vpixels != imdiff->vpixels) {
      perror("LMODEIM: Image panel size changed on reentry. Aborting\n");
      exit(1);
    }
  }

  half_height = imdiff->mode_height / 2;
  half_width = imdiff->mode_width / 2;
  image_length = imdiff->image_length;
  hpixels = imdiff->hpixels;
  vpixels = imdiff->vpixels;

  if (reentry == 0 || reentry == 1) {

    //    printf("MODEIM: Allocating arrays\n");

    num_per_jblock = 0;
    for (jblock = 0; jblock < num_jblocks; jblock++) {
      size_t jlo = half_height + jblock;
      size_t jhi = vpixels - half_height;     
      size_t num_this = (jhi - jlo)/num_jblocks+1;
      if (num_per_jblock < num_this) num_per_jblock = num_this;
    }
    num_per_iblock = 0;
    for (iblock = 0; iblock < num_iblocks; iblock++) {
      size_t ilo = half_width + iblock;
      size_t ihi = hpixels - half_width;     
      size_t num_this = (ihi - ilo)/num_iblocks+1;
      if (num_per_iblock < num_this) num_per_iblock = num_this;
    }

    image = (IMAGE_DATA_TYPE *)calloc(image_length,sizeof(IMAGE_DATA_TYPE));
    image_mode = (size_t *)calloc(image_length,sizeof(size_t));
    nvals = (size_t *)calloc(num_per_jblock*num_per_iblock,sizeof(size_t));
    window = (size_t *)calloc(wlen*num_per_jblock*num_per_iblock,sizeof(size_t));
    stack = (size_t *)calloc(wlen*num_per_jblock*num_per_iblock,sizeof(size_t));

    /* 
     * Allocate working mode filetered image: 
     */ 
  
    if (!image || !image_mode || !window || !stack) {
      sprintf(imdiff->error_msg,"\nLMODEIM:  Couldn't allocate arrays.\n\n");
      return_value = 1;
      goto CloseShop;
    }

#ifdef USE_OFFLOAD
#pragma omp target enter data map(alloc:image[0:image_length],image_mode[0:image_length])
#pragma omp target enter data map(alloc:window[0:wlen*num_per_jblock*num_per_iblock],stack[0:wlen*num_per_jblock*num_per_iblock],nvals[0:num_per_jblock*num_per_iblock])
#endif

  } 

  //  printf("LMODEIM: mode_height = %d, mode_width = %d, num_panels = %d\n",imdiff_in->mode_height,imdiff_in->mode_width,imdiff_in->num_panels);

  for (pidx = 0; pidx < imdiff_in->num_panels; pidx++) {
    size_t num_mode_values=0, num_median_values=0, num_med90_values=0, num_this_values=0, num_ignored_values=0;
    imdiff = &imdiff_in[pidx];
    if (pidx != imdiff->this_panel) {
      perror("LMODEIM: Image panels are not indexed sequentially. Aborting\n");
      exit(1);
    }
    if (hpixels != imdiff->hpixels || vpixels != imdiff->vpixels) {
      perror("LMODEIM: Image panels are not identically formatted. Aborting\n");
      exit(1);
    }

    memcpy(image, imdiff->image, image_length*sizeof(IMAGE_DATA_TYPE));

    IMAGE_DATA_TYPE overload_tag = imdiff->overload_tag;
    IMAGE_DATA_TYPE ignore_tag = imdiff->ignore_tag;
    

    // Compute min and max for image

    int got_first_val = 0;
    
    size_t index; 

    for (index = 0; index < image_length; index++) {
      if ((image[index] != overload_tag) &&
	  (image[index] != ignore_tag) &&
	  (image[index] < MAX_IMAGE_DATA_VALUE)) {
	if (got_first_val != 0) {
	  if (image[index] < minval) minval = image[index];
	  if (image[index] > maxval) maxval = image[index];
	} else {
	  minval = image[index];
	  maxval = image[index];
	  got_first_val = 1;
	}
      }
    }
    // Allocate the distribution

    binsize = imdiff->mode_binsize;

    // Compute the mode filtered image


#ifdef USE_OFFLOAD
#pragma omp target update to(image[0:image_length],image_mode[0:image_length])
#endif

    double start = ltime();

    tic = ltime();

    for (jblock = 0; jblock < num_jblocks; jblock++) {
      for (iblock = 0; iblock < num_iblocks; iblock++) {
	size_t jlo = half_height + jblock;
	size_t jhi = vpixels - half_height;     
	size_t ilo = half_width + iblock;
	size_t ihi = hpixels - half_width;     
#ifdef USE_OPENMP
#ifdef USE_OFFLOAD
#pragma omp target map(to:minval,binsize,wlen,num_jblocks,num_iblocks, \
		       num_per_jblock,num_per_iblock,jlo,jhi,ilo,ihi) 
#pragma omp teams distribute parallel for collapse(2) schedule(static,1)
#else
#pragma omp parallel for shared(stack,window,nvals,image,image_mode) \
  private(i,j)
#endif
#endif
      for (j = jlo; j < jhi; j=j+num_jblocks) {
	for (i = ilo; i < ihi; i=i+num_iblocks) {
	    size_t windidx = (((j-jlo)/num_jblocks)*num_per_iblock+(i-ilo)/num_iblocks);
	    size_t *this_window = &window[windidx*wlen];
            size_t *this_stack = &stack[windidx*wlen];
	    size_t *this_nval = &nvals[windidx];
	    size_t k;
	    for (k = 0; k < wlen; k++) {
	      this_window[k] = ULONG_MAX;
	    }
	    size_t l = 0;
	    size_t wind = 0;
	    RCCOORDS_DATA r, c, rlo, rhi, clo, chi;
	    rlo = j - half_height;
	    rhi = (j + half_height);
	    clo = i - half_width;
	    chi = (i + half_width);
	    for (r = rlo; r <= rhi; r++) {
	      for (c = clo; c <= chi; c++) {
		size_t index = r*hpixels + c;
		if ((image[index] != overload_tag) &&
		    (image[index] != ignore_tag) &&
		    (image[index] < MAX_IMAGE_DATA_VALUE)) {
		  this_window[wind] = (image[index]-minval)/binsize + 1;
		  l++;
		}
		else {
		  this_window[wind] = ULONG_MAX;
		}
		wind++;
	      }
	    }
	    *this_nval = l;
	}
      }

      toc = ltime();
      tmkarr = toc - tic;

      //#ifdef USE_OFFLOAD
      //#pragma omp target map(to:wlen,num_per_jblock,num_per_iblock)  
      //#endif
      tic = ltime();
#ifdef USE_CUDA
      quickSortListCUB(window,stack,num_per_iblock*num_per_jblock,wlen);
#else
      quickSortList(window,stack,num_per_iblock*num_per_jblock,wlen);
#endif
      toc = ltime();
      tsort = toc - tic;

	/*
#ifdef USE_OPENMP
#ifdef USE_OFFLOAD
#pragma omp teams distribute parallel for schedule(static,1)
#else
#pragma omp parallel for shared(stack,window) 
#endif
#endif
      for (i = 0; i < num_per_iblock*num_per_jblock; i++) {
	size_t *this_window = &window[i*wlen];
	size_t *this_stack = &stack[i*wlen];
	quickSortIterative(this_window,this_stack,0,wlen-1);
      }
	*/
	
      /*
#ifdef USE_OPENMP
#ifdef USE_OFFLOAD
#pragma omp target map(to:minval,binsize,wlen,overload_tag,ignore_tag,num_jblocks,num_iblocks, \
		       num_per_jblock,num_per_iblock,jlo,jhi,ilo,ihi)	\
        map(tofrom:num_mode_values, num_median_values, num_med90_values, num_this_values,num_ignored_values)
#pragma omp teams distribute parallel for collapse(2) schedule(static,1) \
  reduction(+:num_mode_values, num_median_values, num_med90_values, num_this_values,num_ignored_values)
#else
    //#pragma omp distribute parallel for collapse(2)
#pragma omp parallel for shared(stack,window,nvals,image,image_mode)	\
  reduction(+:num_mode_values, num_median_values, num_med90_values, num_this_values,num_ignored_values)
#endif
#endif
      */
      tic = ltime();
#ifdef USE_OPENMP
#pragma omp parallel for default(shared) \
  private(i,j) \
  reduction(+:num_mode_values, num_median_values, num_med90_values, num_this_values,num_ignored_values)
#endif
      for (j = jlo; j < jhi; j=j+num_jblocks) {
	for (i = ilo; i < ihi; i=i+num_iblocks) {
	    int mode_ct = 0;
	    size_t mode_value=0, max_count=0;
	    size_t index_mode = j*hpixels + i;
	    size_t this_value = (image[index_mode]-minval)/binsize + 1;
	    size_t windidx = (((j-jlo)/num_jblocks)*num_per_iblock+(i-ilo)/num_iblocks);
	    size_t *this_window = &window[windidx*wlen];
  	    size_t *this_stack = &stack[windidx*wlen];
	    size_t *this_nval = &nvals[windidx];
	    size_t l = *this_nval;
	    if (l == 0 || image[index_mode] == ignore_tag || image[index_mode] == overload_tag || image[index_mode] >= MAX_IMAGE_DATA_VALUE) {
	      image_mode[index_mode] = 0;
	      num_ignored_values++;
	    }
	    else {
	      //          printf("Starting quicksort for i=%d,j=%ld\n",i,index_mode/hpixels);
	      //	  insertion_sort(this_window,0,l-1);
	      //	  quicksort(this_window,0,l-1);
	      //          printf("Done with quicksort for i=%d,j=%ld\n",i,index_mode/hpixels);
	      // Get the median
  	      int kmed = l/2;
	      int k90 = l*9/10;
	      size_t min_value = this_window[0];
	      size_t max_value = this_window[l-1];
	      size_t range_value = max_value - min_value;
  	      size_t median_value = this_window[kmed];
	      size_t med90_value = this_window[k90];

	      // Get the mode
	      size_t this_count = 1;
	      size_t last_value = this_window[0];
	      max_count = 1;
	      size_t k;
	      for (k = 1; k < l; k++) {
		if (this_window[k] == last_value) {
		  this_count++;
		} else {
		  last_value = this_window[k];
		  this_count = 1;
		}
		if (this_count > max_count) max_count = this_count;
	      }
	      this_count = 1;
	      last_value = this_window[0];
	      double p, entropy = 0.0;
	      for (k = 1; k < l; k++) {
		if (this_window[k] == last_value) {
		  this_count++;
		} else {
		  p = (double)this_count/(double)l;
		  entropy -=  p * log_local(p);
		  last_value = this_window[k];
		  this_count = 1;
		}
		if (this_count == max_count) {
		  mode_value += this_window[k];
		  mode_ct++;
		}
	      }
	      mode_value = (size_t)(((float)mode_value/(float)mode_ct) + .5);
	      p = (double)this_count/(double)l;
	      entropy -=  p * log_local(p);
	      //	  image_mode[index_mode] = (size_t)(((float)mode_value/(float)mode_ct) + .5);
#ifdef DEBUG
	      if (j == 600 && i == 600) {
		printf("LMODEIM: entropy = %g, mode_ct = %d, mode_value = %ld, median_value = %ld, range_value = %ld, this_value = %ld, med90_value = %ld, kmed = %d, k90 = %d\n",entropy,mode_ct,mode_value,median_value,range_value,this_value,med90_value,kmed,k90);
	      } 
#endif 
	      // Depending on the distribution, use alternative values other than the mode
		if (range_value <= 2) {
		  image_mode[index_mode]  = this_value;
		  num_this_values++;
		} else {
		  if (this_value <= med90_value) {
		    image_mode[index_mode] = this_value;
		    num_this_values++;
		  } else {
		    image_mode[index_mode] = this_window[(size_t)(((double)k90*(double)rand_local())/(double)ULONG_MAX)];
		    num_med90_values++;
		  //	      printf("%d %ld %ld\n",kmed,mode_value,median_value);
		  //	      mode_value = median_value;
		  }
		}
		/*	      if (entropy > log(10.)) {
		if (mode_ct == 1) {
		  image_mode[index_mode] = median_value;
		  num_median_values++;
		} else {
		  image_mode[index_mode] = mode_value;
		  num_mode_values++;
		}
	      } else {
		if (range_value <= 2) {
		  image_mode[index_mode]  = this_value;
		  num_this_values++;
		} else {
		  if (this_value <= med90_value) {
		    image_mode[index_mode] = this_value;
		    num_this_values++;
		  } else {
		    image_mode[index_mode] = this_window[(size_t)(((double)k90*(double)rand())/(double)RAND_MAX)];
		    num_med90_values++;
		  //	      printf("%d %ld %ld\n",kmed,mode_value,median_value);
		  //	      mode_value = median_value;
		  }
		}
	      }
		*/
	    }
	    //        printf("Stop tm = %ld,th = %ld,i = %d,j = %d\n",tm,th,i,j);
	  }
	}
      toc = ltime();
      tmkimg = toc - tic;
#ifdef DEBUG
      printf("iblock = %d, jblock = %d, tmkarr = %g secs, tsort = %g secs, tmkimg = %g secs\n",iblock,jblock,tmkarr,tsort,tmkimg);
#endif

      }
    }

    // Now image_mode holds the mode filtered values
    // Convert these values to pixel values and store them in the input image

    double stop = ltime();

    double tel = stop-start;
    fflush(stdout);

#ifdef DEBUG
    printf("LMODEIM: %g seconds, num_mode_values=%ld,num_median_values=%ld,num_this_values=%ld,num_med90_values=%ld,num_ignored_values=%ld\n",tel,num_mode_values,num_median_values,num_this_values,num_med90_values,num_ignored_values);
#endif
    
#ifdef USE_OPENMP
#ifdef USE_OFFLOAD
#pragma omp target teams distribute map(to:minval,binsize,ignore_tag)
#else
#pragma omp distribute
#endif
#endif

    for (j = 0; j < vpixels; j++) {
      if (j < half_height || j > (vpixels-half_height)) {
	for (i = 0; i < hpixels; i++) {
	  size_t this_index = j * hpixels + i;
	  image[this_index] = ignore_tag;
	} 
      } else {

	for (i = 0; i < half_width; i++) {
	  size_t this_index = j * hpixels + i;
	  image[this_index] = ignore_tag;
	}
	
	for (i = hpixels - half_width; i < hpixels; i++) {
	  size_t this_index = j * hpixels + i;
	  image[this_index] = ignore_tag;
	}

#ifdef USE_OPENMP
#ifdef USE_OFFLOAD
#pragma omp parallel for schedule(static,1)
#else
#pragma omp parallel for
#endif
#endif

	for (i = half_width; i < hpixels - half_width; i++) {
	  size_t this_index = j * hpixels + i;
	  if (image_mode[this_index] != 0) {
	    image[this_index] = (image_mode[this_index]-1)*binsize + minval;
	  } else {
	    image[this_index] = ignore_tag;
	  }
	}
      }
    }
	  
#ifdef USE_OFFLOAD
#pragma omp target update from(image[0:image_length]) 
#endif
    memcpy(imdiff->image,image,image_length*sizeof(IMAGE_DATA_TYPE));

    if (reentry == 0 || reentry == 3) {

#ifdef USE_OFFLOAD
#pragma omp target exit data map(delete:image[0:image_length],image_mode[0:image_length],window[0:wlen*num_per_iblock*num_per_jblock],stack[0:wlen*num_per_iblock*num_per_jblock])
#endif

      //      printf("MODEIM: Freeing arrays\n");
      free(image);
      free(image_mode);
      free(window);
      free(nvals);
      free(stack);
    }
  }
 CloseShop:
  return(return_value);
}

int lmodeim_old(DIFFIMAGE *imdiff_in) 
{
  
  RCCOORDS_DATA 
    half_height,
    half_width,
    n, 
    m, 
    r, 
    c; 

  size_t
    avg_max_count_count = 0,
    num_max_count_1 = 0,
    index = 0; 
  
  float
    avg_max_count = 0;

  IMAGE_DATA_TYPE
    *image;

  int
    return_value = 0;

  int
    pidx,
    nt;
  
  DIFFIMAGE *imdiff;

  for (pidx = 0; pidx < imdiff_in->num_panels; pidx++) {
    imdiff = &imdiff_in[pidx];
    if (pidx != imdiff->this_panel) {
      perror("LMODEIM: Image panels are not indexed sequentially. Aborting\n");
      exit(1);
    }

    /* 
     * Allocate working image: 
     */ 
  
    image = (IMAGE_DATA_TYPE *)calloc(imdiff->image_length, 
				      sizeof(IMAGE_DATA_TYPE)); 
    if (!image) {
      sprintf(imdiff->error_msg,"\nLMODEIM:  Couldn't allocate arrays.\n\n");
      return_value = 1;
      goto CloseShop;
    }


    half_height = imdiff->mode_height / 2;
    half_width = imdiff->mode_width / 2;
    index = 0;

    size_t j;

#ifdef USE_OPENMP
    //  omp_set_num_threads(16);
    //  nt = omp_get_max_threads();
    //  printf("Using OpenMP with %d threads\n",nt);
    {
      //    #pragma omp for schedule(dynamic,1)
      //#pragma omp parallel for shared(imdiff,image,half_height,half_width) private(j) reduction(+:avg_max_count,avg_max_count_count,num_max_count_1)
#pragma omp parallel for shared(imdiff,image,half_height,half_width) private(j)
#endif
      for (j=0; j<imdiff->vpixels; j++) {
	size_t i;
	size_t *count;
	count = (size_t *)calloc(MAX_IMAGE_DATA_VALUE,sizeof(size_t));
	unsigned int *count_pointer;
	count_pointer = (unsigned int *)calloc((imdiff->mode_height+1) *
					       (imdiff->mode_width+1), 
					       sizeof(unsigned int));
	for (i=0; i<imdiff->hpixels; i++) {
	  long mode_value=0;
	  size_t max_count=0;
	  size_t mode_ct=1;
	  size_t l=0;
	  RCCOORDS_DATA n,m,r,c;
	  size_t k;
	  size_t index;
	  index = j*imdiff->hpixels+i;
	  if (imdiff->image[index] != imdiff->ignore_tag && imdiff->image[index] != imdiff->overload_tag) {
	    for(n=-half_height; n<=half_height; n++) {
	      r = j + n;
	      for(m=-half_width; m<=half_width; m++) {
		c = i + m;
		if ((r >= 0) && (r < imdiff->vpixels) && (c >= 0) &&       
		    (c < imdiff->hpixels)) {
		  size_t imd_index;
		  imd_index = index + n*imdiff->hpixels + m;
		  if ((imdiff->image[imd_index] != imdiff->overload_tag) &&
		      (imdiff->image[imd_index] != imdiff->ignore_tag) &&
		      (imdiff->image[imd_index] < MAX_IMAGE_DATA_VALUE) &&
		      (imdiff->image[imd_index] >= 0)) {
		    //		count_pointer[l]=(imdiff->image[imd_index] - 
		    //				    (imdiff->image[imd_index] % 
		    //				     imdiff->mode_binsize) + 32768);		
		    count_pointer[l]=(unsigned int)imdiff->image[imd_index];
		    //		l++;
		    count[count_pointer[l]]++;
		    l++;
		  }
		}
	      }
	    }
	    if (l==0) {
	      image[index]=imdiff->ignore_tag;
	    }
	    else {
	      //	  for(k=0;k<l;k++) {
	      //	    count[count_pointer[k]]++;
	      //	  }
	      for(k=0;k<l;k++) {
		if (count[count_pointer[k]] == max_count) {
		  mode_value += count_pointer[k];
		  mode_ct++;
		}
		else if (count[count_pointer[k]] > max_count) {
		  mode_value = count_pointer[k];
		  max_count = count[count_pointer[k]];
		  mode_ct = 1;
		}
	      }
	      for(k=0;k<l;k++) count[count_pointer[k]] = 0;
	      image[index] = (IMAGE_DATA_TYPE)((float)mode_value/(float)mode_ct);
	      //	  avg_max_count += max_count;
	      //	  avg_max_count = (avg_max_count*avg_max_count_count +
	      // max_count) / (float)(avg_max_count_count + 1);
	      //	  if (max_count == 1) {num_max_count_1++;}
	      //	  avg_max_count_count++;
	    }
	  }
	  else {
	    image[index] = imdiff->image[index];
	  }
	  //      index++;
	  //      if (index % 1000 == 0) printf("%d\n",index);
	}
	free(count);
	free(count_pointer);
      }
#ifdef USE_OPENMP
    }
#endif
    for(index=0;index<imdiff->image_length; index++) {
      imdiff->image[index] = image[index];
    }
    //avg_max_count /= (float)avg_max_count_count;
    //  printf("avg_max_count,num_max_count_1 = %f,%d\n\n",avg_max_count,num_max_count_1);/***/
    free(image);/***/
  } // for(pidx)

 CloseShop:
  return(return_value);
}
