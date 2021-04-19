%{";"
#include <iostream>
#include <string>
#include <stdlib.h>
#include <map>
#include "proj_types.h"
#include "Expression.tab.hpp"
SelectQuery* process_query(std::string);
//stores the value of the tokens.
%}
/*Flex definitions*/
value ^(-?)(0|([1-9][0-9]*))(\\.[0-9]+)?$ 
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
{identifier} {yylval.identifier = new std::string(yytext); return Identifier;}
%%

void show(SelectQuery* s)
{
	for(auto it: s->select_columns)
		std::cout<<it<<", ";
	std::cout<<s->select_expression<<' ';
	for(auto it: s->order_term)
		std::cout<<it.first<<" "<<it.second<<", ";
    return;
}
int yywrap(void) {
    return 1;
}
