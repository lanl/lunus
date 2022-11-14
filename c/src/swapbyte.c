#include<stdio.h>
#include<stdlib.h>

int main()
{
	char ch1,
		ch2;

	while( fread(&ch1, 1, 1, stdin) == 1) {
	  fread(&ch2, 1, 1, stdin);
	  fwrite(&ch2, 1, 1, stdout);
	  fwrite(&ch1, 1, 1, stdout);
	}
}
