/* Example sobel code for ECE574 -- Fall 2015 */
/* By Vince Weaver <vincent.weaver@maine.edu> */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>

#include <jpeglib.h>

#include <mpi.h>

/* Filters */
static int sobel_x_filter[3][3]={{-1,0,+1},{-2,0,+2},{-1,0,+1}};
static int sobel_y_filter[3][3]={{-1,-2,-1},{0,0,0},{1,2,+1}};

/* Structure describing the image */
struct image_t {
	int x;
	int y;
	int depth;	/* bytes */
	unsigned char *pixels;
};

struct convolve_data_t {
	struct image_t *old;
	struct image_t *new;
	int (*filter)[3][3];
	int ystart;
	int yend;
};


/* very inefficient convolve code */
static void *generic_convolve(void *argument) {

	int x,y,k,l,d;
	uint32_t color;
	int sum,depth,width;

	struct image_t *old;
	struct image_t *new;
	int (*filter)[3][3];
	struct convolve_data_t *data;
	int ystart, yend, ystart_original;

	/* Convert from void pointer to the actual data type */
	data=(struct convolve_data_t *)argument;
	old=data->old;
	new=data->new;
	filter=data->filter;

	ystart=data->ystart;
	
	// FIXED bounds
	ystart_original = ystart;
	if(ystart == 0) ystart = 1;

	yend=data->yend;
	if(yend==old->y) yend--;

	depth=old->depth;
	width=old->x*old->depth;

//	if (ystart==0) ystart=1;
//	if (yend==old->y) yend=old->y-1;

	for(d=0;d<3;d++) {
	   for(x=1;x<old->x-1;x++) {
	     for(y=(ystart);y<(yend);y++) {
		sum=0;
		for(k=-1;k<2;k++) {
		   for(l=-1;l<2;l++) {
			color=old->pixels[((y+l)*width)+(x*depth+d+k*depth)];
			sum+=color * (*filter)[k+1][l+1];
		   }
		}

		if (sum<0) sum=0;
		if (sum>255) sum=255;
	
		// y-ystart to start at offset 0
		new->pixels[((y-(ystart_original))*width)+x*depth+d]=sum;
	     }
	   }
	}


	return NULL;
}


static int load_jpeg(char *filename, struct image_t *image) {

	FILE *fff;
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	JSAMPROW output_data;
	unsigned int scanline_len;
	int scanline_count=0;

	fff=fopen(filename,"rb");
	if (fff==NULL) {
		fprintf(stderr, "Could not load %s: %s\n",
			filename, strerror(errno));
		return -1;
	}

	/* set up jpeg error routines */
	cinfo.err = jpeg_std_error(&jerr);

	/* Initialize cinfo */
	jpeg_create_decompress(&cinfo);

	/* Set input file */
	jpeg_stdio_src(&cinfo, fff);

	/* read header */
	jpeg_read_header(&cinfo, TRUE);

	/* Start decompressor */
	jpeg_start_decompress(&cinfo);

	printf("output_width=%d, output_height=%d, output_components=%d\n",
		cinfo.output_width,
		cinfo.output_height,
		cinfo.output_components);

	image->x=cinfo.output_width;
	image->y=cinfo.output_height;
	image->depth=cinfo.output_components;

	scanline_len = cinfo.output_width * cinfo.output_components;
	image->pixels=malloc(cinfo.output_width * cinfo.output_height * cinfo.output_components);

	while (scanline_count < cinfo.output_height) {
		output_data = (image->pixels + (scanline_count * scanline_len));
		jpeg_read_scanlines(&cinfo, &output_data, 1);
		scanline_count++;
	}

	/* Finish decompressing */
	jpeg_finish_decompress(&cinfo);

	jpeg_destroy_decompress(&cinfo);

	fclose(fff);

	return 0;
}

static int store_jpeg(char *filename, struct image_t *image) {

	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	int quality=90; /* % */
	int i;

	FILE *fff;

	JSAMPROW row_pointer[1];
	int row_stride;

	/* setup error handler */
	cinfo.err = jpeg_std_error(&jerr);

	/* initialize jpeg compression object */
	jpeg_create_compress(&cinfo);

	/* Open file */
	fff = fopen(filename, "wb");
	if (fff==NULL) {
		fprintf(stderr, "can't open %s: %s\n",
			filename,strerror(errno));
		return -1;
	}

	jpeg_stdio_dest(&cinfo, fff);

	/* Set compression parameters */
	cinfo.image_width = image->x;
	cinfo.image_height = image->y;
	cinfo.input_components = image->depth;
	cinfo.in_color_space = JCS_RGB;
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, quality, TRUE);

	/* start compressing */
	jpeg_start_compress(&cinfo, TRUE);

	row_stride=image->x*image->depth;

	for(i=0;i<image->y;i++) {
		row_pointer[0] = & image->pixels[i * row_stride];
		jpeg_write_scanlines(&cinfo, row_pointer, 1);
	}

	/* finish compressing */
	jpeg_finish_compress(&cinfo);

	/* close file */
	fclose(fff);

	/* clean up */
	jpeg_destroy_compress(&cinfo);

	return 0;
}

static int combine(struct image_t *s_x,	struct image_t *s_y, struct image_t *new) {
	int i;
	int out;

	for(i=0;i<( s_x->depth * s_x->x * s_x->y );i++) {

		out=sqrt(
			(s_x->pixels[i]*s_x->pixels[i])+
			(s_y->pixels[i]*s_y->pixels[i])
			);

		if (out>255) out=255;
		if (out<0) out=0;
		new->pixels[i]=out;
	}

	return 0;
}

int main(int argc, char **argv) {

	struct image_t image,sobel_x,sobel_y,new_image;
	struct convolve_data_t sobel_data[2];
	double start_time,load_time,store_time,convolve_time,combine_time;
	int result;

	int numtasks, rank;

	/* Check command line usage */
	if (argc<2) {
		fprintf(stderr,"Usage: %s image_file\n",argv[0]);
		return -1;
	}

	/* Initialize MPI */
	result = MPI_Init(&argc,&argv);
	if (result != MPI_SUCCESS) {
		printf ("Error starting MPI program!.\n");
		MPI_Abort(MPI_COMM_WORLD, result);
	}

	// Get the rank and size parameters
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	printf("Number of tasks: %d \tMy Rank: %d\n", numtasks, rank);

	// Get the load jpeg time only for rank 0
	if(rank == 0) {

		start_time=MPI_Wtime();

		/* Load an image */
		load_jpeg(argv[1],&image);

		load_time=MPI_Wtime();

	}

	// Load the jpeg in the other ranks
	if(rank != 0) {
		/* Load an image */
		load_jpeg(argv[1],&image);
	}


	/* Allocate space for output image */
	new_image.x=image.x;
	new_image.y=image.y;
	new_image.depth=image.depth;
	new_image.pixels=malloc(image.x*image.y*image.depth*sizeof(char));
	
	/* Allocate space for output image */
	sobel_x.x=image.x;
	sobel_x.y=image.y;
	sobel_x.depth=image.depth;
	sobel_x.pixels=malloc(image.x*image.y*image.depth*sizeof(char));

	/* Allocate space for output image */
	sobel_y.x=image.x;
	sobel_y.y=image.y;
	sobel_y.depth=image.depth;
	sobel_y.pixels=malloc(image.x*image.y*image.depth*sizeof(char));

	/* convolution */
	sobel_data[0].old=&image;
	sobel_data[0].new=&sobel_x;
	sobel_data[0].filter=&sobel_x_filter;

	// Break up the image by the ranks
	sobel_data[0].ystart=((rank*image.y) / numtasks);
	sobel_data[0].yend=(((rank+1)*image.y) / numtasks);
	
	// If the last rank go to the end
	if(rank == (numtasks - 1)) {
		sobel_data[0].yend=image.y;
	}
	generic_convolve((void *)&sobel_data[0]);

	// PRINTF FOR DEBUGGING
	printf("Rank: %d \t ystart: %d\t yend: %d\n", rank, sobel_data[0].ystart, sobel_data[0].yend);

	// SOBEL Y
	sobel_data[1].old=&image;
	sobel_data[1].new=&sobel_y;
	sobel_data[1].filter=&sobel_y_filter;

	// Break up the image by the ranks
	sobel_data[1].ystart=((rank*image.y) / numtasks);
	sobel_data[1].yend=(((rank+1)*image.y) / numtasks);

	// If the last rank, go till the end
	if(rank == (numtasks - 1)) {
		sobel_data[1].yend=image.y;
	}

	// PRINTF FOR DEBUGGING
	generic_convolve((void *)&sobel_data[1]);

	// Only get convolve time for rank 0
	if(rank == 0) convolve_time=MPI_Wtime();

	// Make 2 buffer arrays for the gathers
	char* buffer_x = malloc(image.x*image.y*image.depth*sizeof(char));
	char* buffer_y = malloc(image.x*image.y*image.depth*sizeof(char));

	// Had an actual if statement here at one point...
	// Useless but...
	if(1) {

		// Gather for sobel x
		MPI_Gather(sobel_x.pixels,				// Send buffer
			image.x*(image.y/numtasks)*image.depth,		// Count
			MPI_CHAR,					// Send type
			buffer_x,					// Recv buffer
			image.x*(image.y/numtasks)*image.depth,		// Count
			MPI_CHAR,					// Recv type
			0,						// root
			MPI_COMM_WORLD);				// where to bcast


		printf("RANK: %d \tI MADE IT AFTER SOBEL X\n", rank);

		// Gather for sobel y
		MPI_Gather(sobel_y.pixels,				// Send buffer
			image.x*(image.y/numtasks)*image.depth,		// Count
			MPI_CHAR,					// Send type
			buffer_y,					// Recv buffer
			image.x*(image.y/numtasks)*image.depth,		// Count
			MPI_CHAR,					// Recv type
			0,						// root
			MPI_COMM_WORLD);				// where to bcast
	

		printf("RANK: %d \tI MADE IT AFTER SOBEL Y\n", rank);

	}

	// Only combine and store in rank 0
	if(rank == 0) {
	
		for(int i = 0; i < (image.x*image.y*image.depth); i++){
			sobel_x.pixels[i] = buffer_x[i];
			sobel_y.pixels[i] = buffer_y[i];
		}

		/* Combine to form output */
		combine(&sobel_x,&sobel_y,&new_image);

		combine_time=MPI_Wtime();

		/* Write data back out to disk */
		store_jpeg("out.jpg",&new_image);

		store_time=MPI_Wtime();
	
		// Print times for rank 0 only
		printf("Load time: %lf\n",load_time-start_time);
       		printf("Convolve time: %lf\n",convolve_time-load_time);
     	  	printf("Combine time: %lf\n",combine_time-convolve_time);
       		printf("Store time: %lf\n",store_time-combine_time);
		printf("Total time = %lf\n",store_time-start_time);

	}

	MPI_Finalize();

	return 0;
}
