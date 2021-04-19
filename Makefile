all: parser
	nvcc server.cu car.cu lex.yy.cu Expression.tab.cu proj_types.cu kernel.cu -lrt -lpthread -o app
parser:
	yacc -Wall -d Expression.y -o Expression.tab.cpp
	mv Expression.tab.cpp Expression.tab.cu
	mv Expression.tab.hpp Expression.tab.h
	lex -o lex.yy.cu Expression.l

	
