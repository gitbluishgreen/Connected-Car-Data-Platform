%{";"
#include <iostream>
#include <string>
#include <stdlib.h>
#include <map>
#include "proj_types.cuh"
#include "Expression.tab.cuh"
SelectQuery* process_query(std::string);
//stores the value of the tokens.
%}
/*Flex definitions*/
integer (0|[1-9][0-9]*)
value {integer}([.][0-9]+)?
identifier [a-zA-Z_][0-9a-zA-Z_]*
space "\t"|"\n"|"\r"|"\f"|" "
%% 
{space} {}
{value}  {yylval.value = std::stod(yytext);return Value;}
"+" {return Plus;}
"-"  {return Minus;}
"*" {return Mult;}
"/" {return Div;}
"%" {return Modulo;}
"==" {return DoubleEqual;}
"!=" {return NotEqual;}
"!" {return Not;}
">=" {return GreaterEqual;}
"<=" {return LesserEqual;}
">" {return Greater;}
"<" {return Lesser;}
"(" {return OpeningBracket;}
")" {return ClosingBracket;}
"," {return Comma;}
"&&" {return And;}
"||" {return Or;}
"SELECT" {return Select;}
"DISTINCT" {return Distinct;}
"LIMIT" {return Limit;}
"ORDER" {return Order;}
"BY" {return By;}
"GROUP" {return Group;}
"WHERE" {return Where;}
"ASC" {return Ascending;}
"DESC" {return Descending;}
"MIN" {return Minimum;}
"MAX" {return Maximum;}
"AVERAGE" {return Average;}
"SUM" {return Sum;}
"VARIANCE" {return Variance;}
"STDDEV" {return StandardDeviation;}
"COUNT" {return Count;}
{identifier} {yylval.identifier = new char[1+strlen(yytext)]; strcpy(yylval.identifier,yytext); return Identifier;}
%%

int yywrap(void) {
    return 1;
}
