module addressRAM(
	input [4:0] step,
	output reg re_RAM,
	output reg [12:0] firstaddr, lastaddr
);
parameter picture_size = 0;
parameter convolution_size = 0;

parameter picture_storage_limit = picture_size*picture_size;

parameter conv0 = picture_storage_limit + (4) * convolution_size;
parameter conv1 = picture_storage_limit + (4+16) * convolution_size;
parameter conv2 = picture_storage_limit + (4+16+32) * convolution_size;
parameter conv3 = picture_storage_limit + (4+16+32+64) * convolution_size;
parameter conv4 = picture_storage_limit + (4+16+32+64+128) * convolution_size;
parameter conv5 = picture_storage_limit + (4+16+32+64+128+256) * convolution_size;

parameter dense0 =  conv5 + 176;


always @(step)
case (step)
1'd1: begin       //picture
		firstaddr = 0;
		lastaddr = picture_storage_limit;
		re_RAM = 1;
	  end 
2: begin       //weights conv0 
		firstaddr = picture_storage_limit;
		lastaddr = conv0;
		re_RAM = 1;
	  end
4: begin       //weights conv1 
		firstaddr = conv0;
		lastaddr = conv1;
		re_RAM = 1;
	  end
6: begin       //weights conv2 
		firstaddr = conv1;
		lastaddr = conv2;
		re_RAM = 1;
	  end
8: begin       //weights conv3 
		firstaddr = conv2;
		lastaddr = conv3;
		re_RAM = 1;
	  end
10: begin       //weights conv4 
		firstaddr = conv3;
		lastaddr = conv4;
		re_RAM = 1;
	  end
12: begin       //weights conv5 
		firstaddr = conv4;
		lastaddr = conv5;
		re_RAM = 1;
	  end
14: begin       //weights dense0 
		firstaddr = conv5;
		lastaddr = dense0;
		re_RAM = 1;
	  end
default:
			begin
				re_RAM = 0;
			end
endcase
endmodule
