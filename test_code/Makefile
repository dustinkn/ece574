CC = mpicc
CFLAGS = -O2 -Wall -g
LFLAGS = -ljpeg -lm 

all:   sobel_before sobel_coarse sobel_fine

sobel_before:	sobel_before.o
		$(CC) -o sobel_before sobel_before.o $(LFLAGS)

sobel_before.o:	sobel_before.c
		$(CC) $(CFLAGS) -c sobel_before.c


sobel_coarse:	sobel_coarse.o
		$(CC) -o sobel_coarse sobel_coarse.o $(LFLAGS)

sobel_coarse.o:	sobel_coarse.c
		$(CC) $(CFLAGS) -c sobel_coarse.c


sobel_fine:	sobel_fine.o
		$(CC) -o sobel_fine sobel_fine.o $(LFLAGS)

sobel_fine.o:	sobel_fine.c
		$(CC) $(CFLAGS) -c sobel_fine.c

clean:
	       rm -f *~ *.o sobel_coarse sobel_fine sobel_before out.jpg


submit:
	tar -czvf hw07_submit.tar.gz Makefile README *.c

