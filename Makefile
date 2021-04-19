objects = Expression.tab.o lex.yy.o kernel.o proj_types.o server.o car.o
all: $(objects)
	nvcc $(objects) -o app
Expression.tab.cpp lex.yy.cpp: parser
%.o: %.cpp
	nvcc -x cu -arch=sm_20 -I. -dc $< -o $@
parser:
	yacc -Wall -d Expression.y -o Expression.tab.cpp
	lex -o lex.yy.cpp Expression.l

	
