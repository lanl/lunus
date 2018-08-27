#include<lunus.h>
#include<cJSON.h>

char * readExptJSON(struct xyzmatrix *a,const char *json_name) {

  char *json_text, *image_name;

  cJSON 
    *j = NULL, 
    *imageset = NULL, 
    *crystal = NULL, 
    *real_space_a = NULL,
    *real_space_b = NULL,
    *real_space_c = NULL;

  if (lreadbuf((void **)&json_text,json_name) == -1) {
    perror("Can't read json file\n");
    goto readExptJSONFail;
  }

  j = cJSON_Parse(json_text);

  imageset = cJSON_GetArrayItem(cJSON_GetObjectItem(j,"imageset"),0);
  
  image_name = cJSON_GetArrayItem(cJSON_GetObjectItem(imageset,"images"),0)->valuestring;

  crystal = cJSON_GetArrayItem(cJSON_GetObjectItem(j,"crystal"),0);

  real_space_a = cJSON_GetObjectItem(crystal,"real_space_a");
  real_space_b = cJSON_GetObjectItem(crystal,"real_space_b");
  real_space_c = cJSON_GetObjectItem(crystal,"real_space_c");

  a->xx = cJSON_GetArrayItem(real_space_a,0)->valuedouble;
  a->xy = cJSON_GetArrayItem(real_space_a,1)->valuedouble;
  a->xz = cJSON_GetArrayItem(real_space_a,2)->valuedouble;
  a->yx = cJSON_GetArrayItem(real_space_b,0)->valuedouble;
  a->yy = cJSON_GetArrayItem(real_space_b,1)->valuedouble;
  a->yz = cJSON_GetArrayItem(real_space_b,2)->valuedouble;
  a->zx = cJSON_GetArrayItem(real_space_c,0)->valuedouble;
  a->zy = cJSON_GetArrayItem(real_space_c,1)->valuedouble;
  a->zz = cJSON_GetArrayItem(real_space_c,2)->valuedouble;

#ifdef DEBUG
  //    printf("Amatrix for image %s: ",image_name);
  //    printf("(%f, %f, %f) ",a->xx,a->xy,a->xz);
  //    printf("(%f, %f, %f) ",a->yx,a->yy,a->yz);
  //    printf("(%f, %f, %f) ",a->zx,a->zy,a->zz);
  //    printf("\n");
#endif		


  return(image_name);

 readExptJSONFail:
  return(NULL);

}

int readAmatrix(struct xyzmatrix *a,const char *amatrix_format,const size_t i) {

  int num_read,str_length;

  char *amatrix_path;

  struct xyzmatrix *amatrix;

  str_length = snprintf(NULL,0,amatrix_format,i);
	    
  amatrix_path = (char *)malloc(str_length+1);
  
  sprintf(amatrix_path,amatrix_format,i);
  
  num_read = lreadbuf((void **)&amatrix,amatrix_path);

  if (num_read != -1) {
  

  // Calculate the transpose of a, to use matrix multiplication method
  
    *a = lmatt(*amatrix);

#ifdef DEBUG
    //    printf("Amatrix for image %d: ",i);
    //    printf("(%f, %f, %f) ",a->xx,a->xy,a->xz);
    //    printf("(%f, %f, %f) ",a->yx,a->yy,a->yz);
    //    printf("(%f, %f, %f) ",a->zx,a->zy,a->zz);
    //    printf("\n");
#endif		

    return(0);
  
  } else {
    return(-1);
  }
}
