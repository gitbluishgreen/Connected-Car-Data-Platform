/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

#ifndef YY_YY_EXPRESSION_TAB_HPP_INCLUDED
# define YY_YY_EXPRESSION_TAB_HPP_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    Plus = 258,
    Minus = 259,
    Mult = 260,
    Div = 261,
    Modulo = 262,
    NotEqual = 263,
    DoubleEqual = 264,
    Greater = 265,
    GreaterEqual = 266,
    Lesser = 267,
    LesserEqual = 268,
    Or = 269,
    And = 270,
    Not = 271,
    Select = 272,
    Distinct = 273,
    Where = 274,
    Order = 275,
    Group = 276,
    By = 277,
    Limit = 278,
    Ascending = 279,
    Descending = 280,
    Comma = 281,
    OpeningBracket = 282,
    ClosingBracket = 283,
    Maximum = 284,
    Minimum = 285,
    Average = 286,
    Variance = 287,
    StandardDeviation = 288,
    Sum = 289,
    Identifier = 290,
    Value = 291
  };
#endif
/* Tokens.  */
#define Plus 258
#define Minus 259
#define Mult 260
#define Div 261
#define Modulo 262
#define NotEqual 263
#define DoubleEqual 264
#define Greater 265
#define GreaterEqual 266
#define Lesser 267
#define LesserEqual 268
#define Or 269
#define And 270
#define Not 271
#define Select 272
#define Distinct 273
#define Where 274
#define Order 275
#define Group 276
#define By 277
#define Limit 278
#define Ascending 279
#define Descending 280
#define Comma 281
#define OpeningBracket 282
#define ClosingBracket 283
#define Maximum 284
#define Minimum 285
#define Average 286
#define Variance 287
#define StandardDeviation 288
#define Sum 289
#define Identifier 290
#define Value 291

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 25 "Expression.y" /* yacc.c:1909  */

	double value;
	char* identifier;
	class SelectQuery* SelectObject;
	bool distinct;
	class ExpressionNode* expression;
	std::vector<char*>* name_list;
	std::vector<std::pair<char*,ExpressionNode*>>* expression_list;
	std::vector<std::pair<ExpressionNode*,bool>>* order_list;

#line 137 "Expression.tab.hpp" /* yacc.c:1909  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);

#endif /* !YY_YY_EXPRESSION_TAB_HPP_INCLUDED  */
