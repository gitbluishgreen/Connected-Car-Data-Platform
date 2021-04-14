%{";"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "P1.tab.hpp"
%}

/*Flex definitions*/
digit [0-9]
integer (0|([1-9]{digit}*))
identifier [a-zA-Z_][0-9a-zA-Z_]*
space "\t"|"\n"|"\r"|"\f"|" "
%% 

"//".* {;}//ignores any comments present in the program. Multiline comments need to be ignored as well.
{integer}  {yylval.integer=atoi(yytext);return Integer;}
{space} {}
"+" {return Plus;}
"-"  {return Minus;}
"*" {return Mult;}
"/" {return Div;}
"%" {return Modulo;}
"!=" {return NotEqual;}
"==" {return DoubleEqual;}
">" {return GreaterThan;}
"<" {return LessThan;}
">=" {return GreaterEqual;}
"<=" {return LessEqual;}
"(" {return OpeningBracket;}
")" {return ClosingBracket;}
"&&" {return And;}
"||" {return Or;}
"!" {return Not;}
","  {return Comma;}
"SELECT" {return Select;}
"WHERE" {return Where;}
"ORDER" {return Order;}
"GROUP" {return Group;} 
"LIMIT" {return Limit;}
"DISTINCT" {return Distinct;}
"BY" {return By;}
"MIN" {return Min;}
"MAX" {return Max;}
"AVERAGE" {return Average;}
"STDDEV" {return StdDev;}
"VARIANCE" {return Variance;}
"COUNT" {return Count;}
"SUM" {return Sum;}
{identifier} {yylval.string = (char*)malloc(strlen(yytext)+1); strcpy(yylval.string,yytext);return Identifier;}
%%
int yywrap(void) {
    return 1;
}
