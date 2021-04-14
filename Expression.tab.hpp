/* A Bison parser, made by GNU Bison 3.5.1.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2020 Free Software Foundation,
   Inc.

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

/* Undocumented macros, especially those whose name start with YY_,
   are private implementation details.  Do not rely on them.  */

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
    Equal = 264,
    Greater = 265,
    GreaterEqual = 266,
    Lesser = 267,
    LesserEqual = 268,
    Or = 269,
    And = 270,
    Not = 271,
    Where = 272,
    Order = 273,
    Group = 274,
    By = 275,
    Limit = 276,
    Distinct = 277,
    Ascending = 278,
    Descending = 279,
    Comma = 280,
    OpeningBracket = 281,
    ClosingBracket = 282,
    Maximum = 283,
    Minimum = 284,
    Average = 285,
    Variance = 286,
    StandardDeviation = 287,
    Count = 288,
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
#define Equal 264
#define Greater 265
#define GreaterEqual 266
#define Lesser 267
#define LesserEqual 268
#define Or 269
#define And 270
#define Not 271
#define Where 272
#define Order 273
#define Group 274
#define By 275
#define Limit 276
#define Distinct 277
#define Ascending 278
#define Descending 279
#define Comma 280
#define OpeningBracket 281
#define ClosingBracket 282
#define Maximum 283
#define Minimum 284
#define Average 285
#define Variance 286
#define StandardDeviation 287
#define Count 288
#define Sum 289
#define Identifier 290
#define Value 291

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 22 "Expression.y"

    double value;
    std::string* identifier;
	class SelectQuery* SelectObject;
	bool distinct;
	class ExpressionNode* expression;
	std::vector<std::string>* name_list;
	std::vector<std::pair<std::string,ExpressionNode*>>* expression_list;
	std::vector<std::pair<ExpressionNode*,bool>>* order_list;

#line 140 "Expression.tab.hpp"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);

#endif /* !YY_YY_EXPRESSION_TAB_HPP_INCLUDED  */
