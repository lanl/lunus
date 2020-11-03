/* LCBFTOOL.C - CBF read and write tools.

  The methods lbufcompress and lbufuncompress are adapted from 
  buffer_based_service.cpp in the DIALS distribution.

  Original author of C++ code is Graeme Winter, Diamond Light Source
  Translated to C for LUNUS by Michael Wall, Los Alamos National Laboratory.

  Following is the DIALS BSD License, which is similar to the Lunus license:

  Copyright (c) 2015 Diamond Light Source, Lawrence Berkeley National Laboratory
  and the Science and Technology Facilities Council, All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  Neither the name of the Diamond Light Source, Lawrence Berkeley National
  Laboratory or the Science and Technology Facilities Council, nor the names of
  its contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/


#include <mwmask.h>


  /* FIXME should add same functions to work for UNSIGNED values */

//Code contributed by Graeme Winter, Diamond Light Source:

typedef union {
  signed char b[2];
  short s;
} u_s;

typedef union {
  signed char b[4];
  int i;
} u_i;

// functions for byte swapping

void byte_swap_short(signed char * b)
{
  signed char c;
  c = b[0];
  b[0] = b[1];
  b[1] = c;
  return;
}

void byte_swap_int(signed char * b)
{
  signed char c;
  c = b[0];
  b[0] = b[3];
  b[3] = c;
  c = b[1];
  b[1] = b[2];
  b[2] = c;
  return;
}

// helper function: is this machine little endian? CBF files are

char little_endian()
{
  int i = 0x1;
  signed char b = ((u_i *) &i)[0].b[0];

  return b;
}


//main functions

// return length of packed array

size_t lbufcompress(const int* values, const size_t sz, signed char *packed)
{
  int current = 0;
  int delta, i;
  unsigned int j;
  signed char le;
  short s;
  signed char c;
  signed char * b;

  le = little_endian();

  size_t n = 0;
  for (j = 0; j < sz; j++)
    {
      delta = values[j] - current;

      if ((-127 <= delta) && (delta < 128))
        {
          c = (signed char) delta;
          packed[n++] = c;
          current += delta;
          continue;
        }

      packed[n++] = -128;

      if ((-32767 <= delta) && (delta < 32768))
        {
          s = (short) delta;
          b = ((u_s *) & s)[0].b;

          if (le==0)
            {
              byte_swap_short(b);
            }

          packed[n++] = b[0];
          packed[n++] = b[1];
          current += delta;
          continue;
        }
      s = -32768;
      b = ((u_s *) & s)[0].b;

      if (le==0)
        {
          byte_swap_short(b);
        }

      packed[n++] = b[0];
      packed[n++] = b[1];

      if ((-2147483647 <= delta) && (delta <= 2147483647))
        {
          i = delta;
          b = ((u_i *) & i)[0].b;

          if (le==0)
            {
              byte_swap_int(b);
            }

          packed[n++] = b[0];
          packed[n++] = b[1];
          packed[n++] = b[2];
          packed[n++] = b[3];
          current += delta;
          continue;
        }

      /* FIXME I should not get here */
      //fail silently or throw an exception?
    }

  return n;
}

// Note: values must be allocated to full size.

void lbufuncompress(const signed char* packed, const size_t packed_sz, int* values, const size_t values_sz)
{
  int current = 0;
  unsigned int j = 0;
  short s;
  signed char c;
  int i;
  signed char le;
  size_t k;

  le = little_endian();

  k = 0;

  //  while (j < packed_sz && k < values_sz)
  while (j < packed_sz)
    //    while (k < values_sz)
    {
      c = packed[j];
      j += 1;

      if (c != -128)
        {
          current += c;
          values[k++]=current;
          continue;
        }

      ((u_s *) & s)[0].b[0] = packed[j];
      ((u_s *) & s)[0].b[1] = packed[j + 1];
      j += 2;

      if (le==0)
        {
          byte_swap_short((signed char *) &s);
        }

      if (s != -32768)
        {
          current += s;
          values[k++]=current;
          continue;
        }

      ((u_i *) & i)[0].b[0] = packed[j];
      ((u_i *) & i)[0].b[1] = packed[j + 1];
      ((u_i *) & i)[0].b[2] = packed[j + 2];
      ((u_i *) & i)[0].b[3] = packed[j + 3];
      j += 4;

      if (le==0)
        {
          byte_swap_int((signed char *) &i);
        }

      current += i;
      values[k++]=current;
    }
  if (k > values_sz) {
    printf("LCBFTOOL: Exceeded length of image, j = %d, k = %d\n",j,k);
    exit(1);
  }
  if (k < values_sz) {
    printf("LCBFTOOL: Did not reach full length of image, j = %d, k = %d\n",j,k);
    exit(1);
  }
}


