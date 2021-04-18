all:
	yacc -Wall -d Expression.y -o Expression.tab.cpp
	lex -o lex.yy.cpp Expression.l
	g++ Expression.tab.cpp lex.yy.cpp proj_types.hpp
