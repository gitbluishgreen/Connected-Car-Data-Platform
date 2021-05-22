/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison implementation for Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.0.4"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* Copy the first part of user declarations.  */
#line 1 "Expression.y" /* yacc.c:339  */

	#include <iostream>
	#include <string>
	#include <string.h>
	#include <stdlib.h>
	#include <utility>
	#include <vector>
	#include <algorithm>
	#include <map>
	#include <cmath>
	#include "proj_types.cuh"
	void yyerror(const char*);
	int yyparse(void);
	int yylex(void);
	int yy_scan_string(const char*);
	int yylex_destroy(void);
	void update_query(SelectQuery*);
	int get_type(char*);
	void update_columns(std::vector<char*>*);
	bool find_column(char*);
	std::map<std::string,int> column_map; 
	SelectQuery* select_query;

#line 90 "Expression.tab.cpp" /* yacc.c:339  */

# ifndef YY_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULLPTR nullptr
#  else
#   define YY_NULLPTR 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* In a future release of Bison, this section will be replaced
   by #include "Expression.tab.hpp".  */
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
#line 25 "Expression.y" /* yacc.c:355  */

	double value;
	char* identifier;
	class SelectQuery* SelectObject;
	bool distinct;
	class ExpressionNode* expression;
	std::vector<char*>* name_list;
	std::vector<std::pair<char*,ExpressionNode*>>* expression_list;
	std::vector<std::pair<ExpressionNode*,bool>>* order_list;

#line 213 "Expression.tab.cpp" /* yacc.c:355  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);

#endif /* !YY_YY_EXPRESSION_TAB_HPP_INCLUDED  */

/* Copy the second part of user declarations.  */

#line 230 "Expression.tab.cpp" /* yacc.c:358  */

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef YY_ATTRIBUTE
# if (defined __GNUC__                                               \
      && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__)))  \
     || defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#  define YY_ATTRIBUTE(Spec) __attribute__(Spec)
# else
#  define YY_ATTRIBUTE(Spec) /* empty */
# endif
#endif

#ifndef YY_ATTRIBUTE_PURE
# define YY_ATTRIBUTE_PURE   YY_ATTRIBUTE ((__pure__))
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# define YY_ATTRIBUTE_UNUSED YY_ATTRIBUTE ((__unused__))
#endif

#if !defined _Noreturn \
     && (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112)
# if defined _MSC_VER && 1200 <= _MSC_VER
#  define _Noreturn __declspec (noreturn)
# else
#  define _Noreturn YY_ATTRIBUTE ((__noreturn__))
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif


#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYSIZE_T yynewbytes;                                            \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / sizeof (*yyptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  14
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   87

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  37
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  20
/* YYNRULES -- Number of rules.  */
#define YYNRULES  53
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  90

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   291

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    46,    46,    51,    66,    78,    85,    89,    95,   100,
     107,   110,   114,   120,   124,   130,   134,   140,   144,   151,
     158,   165,   172,   179,   187,   194,   203,   207,   213,   218,
     224,   228,   234,   240,   252,   265,   277,   282,   299,   315,
     337,   358,   379,   397,   402,   416,   431,   435,   449,   462,
     479,   484,   491,   497
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "Plus", "Minus", "Mult", "Div", "Modulo",
  "NotEqual", "DoubleEqual", "Greater", "GreaterEqual", "Lesser",
  "LesserEqual", "Or", "And", "Not", "Select", "Distinct", "Where",
  "Order", "Group", "By", "Limit", "Ascending", "Descending", "Comma",
  "OpeningBracket", "ClosingBracket", "Maximum", "Minimum", "Average",
  "Variance", "StandardDeviation", "Sum", "Identifier", "Value", "$accept",
  "goal", "Select_Query", "DistinctQualifier", "Columns", "MultiCol",
  "OrderCriteria", "WhereCondition", "LimitExp", "AggregateFunction",
  "AggCol", "MultiAggCol", "GroupExp", "OrderExp", "ExpList", "Exp",
  "Exp1", "Exp2", "Exp3", "Term", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291
};
# endif

#define YYPACT_NINF -20

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-20)))

#define YYTABLE_NINF -1

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int8 yypact[] =
{
     -13,   -15,    20,   -20,   -20,   -20,   -20,   -20,   -20,   -20,
     -20,    -4,     5,    -9,   -20,   -20,   -20,    -9,   -14,   -14,
       9,    13,    16,    10,   -14,   -20,   -20,    -3,    39,    62,
       2,   -20,    61,    31,    16,     6,    35,    44,    39,    12,
      10,    10,   -20,    10,    10,    10,    10,    10,    10,    10,
      10,    10,    10,    10,   -14,    44,   -20,   -14,    43,   -20,
     -20,    39,    39,    54,    62,    62,    62,    62,    62,    62,
       2,     2,   -20,   -20,   -20,    61,   -20,   -14,    19,   -20,
      30,    19,   -20,   -20,   -20,    55,   -20,   -14,    14,   -20
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,     6,     0,     2,     5,    18,    19,    20,    21,    22,
      23,     0,     0,    15,     1,     8,    10,    15,     0,     0,
      28,     7,    30,     0,     0,    51,    52,     0,    36,    43,
      46,    50,    14,     0,    30,     0,     0,    17,    35,     0,
       0,     0,    26,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    17,     9,     0,     0,     3,
      53,    33,    34,    24,    42,    41,    37,    39,    38,    40,
      44,    45,    47,    48,    49,    27,     4,    29,    13,    16,
       0,    13,    11,    12,    32,     0,    31,     0,     0,    25
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
     -20,   -20,   -20,   -20,   -20,   -20,     0,    66,    29,     7,
     -20,   -20,   -20,    51,   -20,   -19,   -17,    26,    28,     3
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
      -1,     2,     3,    11,    17,    21,    84,    20,    59,    12,
      13,    63,    34,    37,    77,    27,    28,    29,    30,    31
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint8 yytable[] =
{
      32,    15,    23,     4,     1,    39,    38,    51,    52,    53,
      19,    40,    41,    24,     5,     6,     7,     8,     9,    10,
      14,    25,    26,    61,    62,    42,    40,    41,    40,    41,
      33,    16,    18,    40,    41,    75,    36,    24,    78,    35,
      60,    56,    89,    82,    83,    25,    26,    43,    44,    45,
      46,    47,    48,    54,    72,    73,    74,    57,    81,     5,
       6,     7,     8,     9,    10,    49,    50,    58,    88,    64,
      65,    66,    67,    68,    69,    40,    41,    70,    71,    79,
      80,    86,    87,    22,    76,    55,     0,    85
};

static const yytype_int8 yycheck[] =
{
      19,     5,    16,    18,    17,    24,    23,     5,     6,     7,
      19,    14,    15,    27,    29,    30,    31,    32,    33,    34,
       0,    35,    36,    40,    41,    28,    14,    15,    14,    15,
      21,    35,    27,    14,    15,    54,    20,    27,    57,    26,
      28,    35,    28,    24,    25,    35,    36,     8,     9,    10,
      11,    12,    13,    22,    51,    52,    53,    22,    77,    29,
      30,    31,    32,    33,    34,     3,     4,    23,    87,    43,
      44,    45,    46,    47,    48,    14,    15,    49,    50,    36,
      26,    81,    27,    17,    55,    34,    -1,    80
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    17,    38,    39,    18,    29,    30,    31,    32,    33,
      34,    40,    46,    47,     0,     5,    35,    41,    27,    19,
      44,    42,    44,    16,    27,    35,    36,    52,    53,    54,
      55,    56,    52,    21,    49,    26,    20,    50,    53,    52,
      14,    15,    28,     8,     9,    10,    11,    12,    13,     3,
       4,     5,     6,     7,    22,    50,    35,    22,    23,    45,
      28,    53,    53,    48,    54,    54,    54,    54,    54,    54,
      55,    55,    56,    56,    56,    52,    45,    51,    52,    36,
      26,    52,    24,    25,    43,    46,    43,    27,    52,    28
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    37,    38,    39,    39,    40,    40,    41,    41,    42,
      42,    43,    43,    43,    44,    44,    45,    45,    46,    46,
      46,    46,    46,    46,    47,    48,    48,    49,    49,    50,
      50,    51,    51,    52,    52,    52,    52,    53,    53,    53,
      53,    53,    53,    53,    54,    54,    54,    55,    55,    55,
      55,    56,    56,    56
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     6,     6,     1,     0,     2,     1,     3,
       0,     1,     1,     0,     2,     0,     2,     0,     1,     1,
       1,     1,     1,     1,     5,     6,     0,     3,     0,     3,
       0,     3,     2,     3,     3,     2,     1,     3,     3,     3,
       3,     3,     3,     1,     3,     3,     1,     3,     3,     3,
       1,     1,     1,     3
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                  \
do                                                              \
  if (yychar == YYEMPTY)                                        \
    {                                                           \
      yychar = (Token);                                         \
      yylval = (Value);                                         \
      YYPOPSTACK (yylen);                                       \
      yystate = *yyssp;                                         \
      goto yybackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;                                                  \
    }                                                           \
while (0)

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256



/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)

/* This macro is provided for backward compatibility. */
#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*----------------------------------------.
| Print this symbol's value on YYOUTPUT.  |
`----------------------------------------*/

static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  FILE *yyo = yyoutput;
  YYUSE (yyo);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
  YYUSE (yytype);
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyoutput, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yytype_int16 *yyssp, YYSTYPE *yyvsp, int yyrule)
{
  unsigned long int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[yyssp[yyi + 1 - yynrhs]],
                       &(yyvsp[(yyi + 1) - (yynrhs)])
                                              );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
yystrlen (const char *yystr)
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            /* Fall through.  */
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                {
                  YYSIZE_T yysize1 = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
                  if (! (yysize <= yysize1
                         && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                    return 2;
                  yysize = yysize1;
                }
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    YYSIZE_T yysize1 = yysize + yystrlen (yyformat);
    if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
      return 2;
    yysize = yysize1;
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
{
  YYUSE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}




/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        YYSTYPE *yyvs1 = yyvs;
        yytype_int16 *yyss1 = yyss;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * sizeof (*yyssp),
                    &yyvs1, yysize * sizeof (*yyvsp),
                    &yystacksize);

        yyss = yyss1;
        yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yytype_int16 *yyss1 = yyss;
        union yyalloc *yyptr =
          (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
                  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 47 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.SelectObject) = (yyvsp[0].SelectObject);
	update_query((yyval.SelectObject));
}
#line 1370 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 3:
#line 52 "Expression.y" /* yacc.c:1646  */
    {
	for(auto it: *(yyvsp[-3].name_list))
	{
		if(!find_column(it))
			YYABORT;
	}
	cudaMallocHost((void**)&(yyval.SelectObject),sizeof(SelectQuery));//pinned memory
	(yyval.SelectObject)->distinct = (yyvsp[-4].distinct);
	std::reverse((yyvsp[-3].name_list)->begin(),(yyvsp[-3].name_list)->end());
	(yyval.SelectObject)->select_columns = (yyvsp[-3].name_list);
	(yyval.SelectObject)->select_expression = (yyvsp[-2].expression);
	(yyval.SelectObject)->order_term = (yyvsp[-1].order_list);
	(yyval.SelectObject)->limit_term = (yyvsp[0].value);
}
#line 1389 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 4:
#line 67 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.SelectObject),sizeof(SelectQuery));
	std::reverse((yyvsp[-4].expression_list)->begin(),(yyvsp[-4].expression_list)->end());
	(yyval.SelectObject)->distinct = false;
	(yyval.SelectObject)->aggregate_columns = (yyvsp[-4].expression_list);
	(yyval.SelectObject)->select_expression =  (yyvsp[-3].expression);
	(yyval.SelectObject)->group_term = (yyvsp[-2].expression);
	(yyval.SelectObject)->order_term = (yyvsp[-1].order_list);
	(yyval.SelectObject)->limit_term = (yyvsp[0].value);  
}
#line 1404 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 5:
#line 79 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.distinct) = true;
}
#line 1412 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 6:
#line 85 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.distinct) = false;
}
#line 1420 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 7:
#line 90 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.name_list) = (yyvsp[0].name_list);
	(yyval.name_list)->push_back((yyvsp[-1].identifier));
}
#line 1429 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 8:
#line 96 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.name_list) = new std::vector<char*>();
	update_columns((yyval.name_list));
}
#line 1438 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 9:
#line 101 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.name_list) = (yyvsp[-2].name_list);
	(yyval.name_list)->push_back((yyvsp[0].identifier));
}
#line 1447 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 10:
#line 107 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.name_list) = new std::vector<char*>();
}
#line 1455 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 11:
#line 111 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.distinct) = false;
}
#line 1463 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 12:
#line 115 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.distinct) = true;
}
#line 1471 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 13:
#line 120 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.distinct) = false;
}
#line 1479 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 14:
#line 125 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.expression) = (yyvsp[0].expression);
}
#line 1487 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 15:
#line 130 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.expression) = NULL;
}
#line 1495 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 16:
#line 135 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.value) = floor((yyvsp[0].value));
}
#line 1503 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 17:
#line 140 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.value) = -1;
}
#line 1511 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 18:
#line 145 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.identifier),4*sizeof(char));
	(yyval.identifier)[0] = 'm';
	(yyval.identifier)[1] = 'a';
	(yyval.identifier)[2] = 'x';
}
#line 1522 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 19:
#line 152 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.identifier),4*sizeof(char));
	(yyval.identifier)[0] = 'm';
	(yyval.identifier)[1] = 'i';
	(yyval.identifier)[2] = 'n';
}
#line 1533 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 20:
#line 159 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.identifier),4*sizeof(char));
	(yyval.identifier)[0] = 'a';
	(yyval.identifier)[1] = 'v';
	(yyval.identifier)[2] = 'g';
}
#line 1544 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 21:
#line 166 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.identifier),4*sizeof(char));
	(yyval.identifier)[0] = 'v';
	(yyval.identifier)[1] = 'a';
	(yyval.identifier)[2] = 'r';
}
#line 1555 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 22:
#line 173 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.identifier),4*sizeof(char));
	(yyval.identifier)[0] = 's';
	(yyval.identifier)[1] = 't';
	(yyval.identifier)[2] = 'd';
}
#line 1566 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 23:
#line 180 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.identifier),4*sizeof(char));
	(yyval.identifier)[0] = 's';
	(yyval.identifier)[1] = 'u';
	(yyval.identifier)[2] = 'm';
}
#line 1577 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 24:
#line 188 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.expression_list) = (yyvsp[0].expression_list);
	if((yyvsp[-2].expression)->type_of_expr == 1)
		YYABORT;//no boolean values allowed.
	(yyval.expression_list)->push_back(std::make_pair((yyvsp[-4].identifier),(yyvsp[-2].expression)));
}
#line 1588 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 25:
#line 195 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.expression_list) = (yyvsp[-5].expression_list);
	if((yyvsp[-1].expression)->type_of_expr == 1)
		YYABORT;//no boolean value allowed.
	(yyval.expression_list)->push_back(std::make_pair((yyvsp[-3].identifier),(yyvsp[-1].expression)));
}
#line 1599 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 26:
#line 203 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.expression_list) = new std::vector<std::pair<char*,ExpressionNode*>>;//do we need to replace by thrust?
}
#line 1607 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 27:
#line 208 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.expression) = (yyvsp[0].expression);
}
#line 1615 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 28:
#line 213 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.expression) = NULL;
}
#line 1623 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 29:
#line 219 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.order_list) = (yyvsp[0].order_list);
}
#line 1631 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 30:
#line 224 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.order_list) = NULL;
}
#line 1639 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 31:
#line 229 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.order_list) = (yyvsp[-2].order_list);
	(yyval.order_list)->push_back(std::make_pair((yyvsp[-1].expression),(yyvsp[0].distinct)));
}
#line 1648 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 32:
#line 235 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.order_list) = new std::vector<std::pair<ExpressionNode*,bool>>();//replace by thrust?
	(yyval.order_list)->push_back(std::make_pair((yyvsp[-1].expression),(yyvsp[0].distinct)));
}
#line 1657 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 33:
#line 241 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.expression),sizeof(ExpressionNode));
	cudaMallocHost((void**)(&((yyval.expression)->exp_operator)),3 * sizeof(char));
	(yyval.expression)->exp_operator[0] = 'o';
	(yyval.expression)->exp_operator[1] = 'r';
	(yyval.expression)->left_hand_term = (yyvsp[-2].expression);
	(yyval.expression)->right_hand_term = (yyvsp[0].expression);
	if((yyvsp[-2].expression)->type_of_expr != 1 || (yyvsp[0].expression)->type_of_expr!= 1)
		YYABORT;
	(yyval.expression)->type_of_expr =  1;
}
#line 1673 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 34:
#line 253 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.expression),sizeof(ExpressionNode));
	cudaMallocHost((void**)(&((yyval.expression)->exp_operator)),4 * sizeof(char));
	(yyval.expression)->exp_operator[0] = 'a';
	(yyval.expression)->exp_operator[1] = 'n';
	(yyval.expression)->exp_operator[2] = 'd';
	(yyval.expression)->left_hand_term = (yyvsp[-2].expression);
	(yyval.expression)->right_hand_term = (yyvsp[0].expression);
	if((yyvsp[-2].expression)->type_of_expr != 1 || (yyvsp[0].expression)->type_of_expr!= 1)
		YYABORT;
	(yyval.expression)->type_of_expr =  1;
}
#line 1690 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 35:
#line 266 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.expression),sizeof(ExpressionNode));
	cudaMallocHost((void**)(&((yyval.expression)->exp_operator)),4 * sizeof(char));
	(yyval.expression)->exp_operator[0] = 'n';
	(yyval.expression)->exp_operator[1] = 'o';
	(yyval.expression)->exp_operator[2] = 't';
	(yyval.expression)->left_hand_term = (yyvsp[0].expression);
	if((yyvsp[0].expression)->type_of_expr != 1)
		YYABORT;
	(yyval.expression)->type_of_expr =  1;
}
#line 1706 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 36:
#line 278 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.expression)=(yyvsp[0].expression);
}
#line 1714 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 37:
#line 283 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.expression),sizeof(ExpressionNode));
	cudaMallocHost((void**)(&((yyval.expression)->exp_operator)),8 * sizeof(char));
	(yyval.expression)->left_hand_term = (yyvsp[-2].expression);
	(yyval.expression)->right_hand_term = (yyvsp[0].expression);
	(yyval.expression)->exp_operator[0] = 'g';
	(yyval.expression)->exp_operator[1] = 'r';
	(yyval.expression)->exp_operator[2] = 'e';
	(yyval.expression)->exp_operator[3] = 'a';
	(yyval.expression)->exp_operator[4] = 't';
	(yyval.expression)->exp_operator[5] = 'e';
	(yyval.expression)->exp_operator[6] = 'r';
	if((yyvsp[-2].expression)->type_of_expr ==  1 || (yyvsp[0].expression)->type_of_expr == 1)
		YYABORT;
	(yyval.expression)->type_of_expr =  1;
}
#line 1735 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 38:
#line 300 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.expression),sizeof(ExpressionNode));
	cudaMallocHost((void**)(&((yyval.expression)->exp_operator)),7 * sizeof(char));
	(yyval.expression)->left_hand_term = (yyvsp[-2].expression);
	(yyval.expression)->right_hand_term = (yyvsp[0].expression);
	(yyval.expression)->exp_operator[0] = 'l';
	(yyval.expression)->exp_operator[1] = 'e';
	(yyval.expression)->exp_operator[2] = 's';
	(yyval.expression)->exp_operator[3] = 's';
	(yyval.expression)->exp_operator[4] = 'e';
	(yyval.expression)->exp_operator[5] = 'r';
	if((yyvsp[-2].expression)->type_of_expr ==  1 || (yyvsp[0].expression)->type_of_expr == 1)
		YYABORT;
	(yyval.expression)->type_of_expr =  1;
}
#line 1755 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 39:
#line 316 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.expression),sizeof(ExpressionNode));
	cudaMallocHost((void**)(&((yyval.expression)->exp_operator)),13 * sizeof(char));
	(yyval.expression)->left_hand_term = (yyvsp[-2].expression);
	(yyval.expression)->right_hand_term = (yyvsp[0].expression);
	(yyval.expression)->exp_operator[0] = 'g';
	(yyval.expression)->exp_operator[1] = 'r';
	(yyval.expression)->exp_operator[2] = 'e';
	(yyval.expression)->exp_operator[3] = 'a';
	(yyval.expression)->exp_operator[4] = 't';
	(yyval.expression)->exp_operator[5] = 'e';
	(yyval.expression)->exp_operator[6] = 'r';
	(yyval.expression)->exp_operator[7] = 'e';
	(yyval.expression)->exp_operator[8] = 'q';
	(yyval.expression)->exp_operator[9] = 'u';
	(yyval.expression)->exp_operator[10] = 'a';
	(yyval.expression)->exp_operator[11] = 'l';
	if((yyvsp[-2].expression)->type_of_expr ==  1 || (yyvsp[0].expression)->type_of_expr == 1)
		YYABORT;
	(yyval.expression)->type_of_expr =  1;
}
#line 1781 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 40:
#line 338 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.expression),sizeof(ExpressionNode));
	cudaMallocHost((void**)(&((yyval.expression)->exp_operator)),12 * sizeof(char));
	(yyval.expression)->left_hand_term = (yyvsp[-2].expression);
	(yyval.expression)->right_hand_term = (yyvsp[0].expression);
	(yyval.expression)->exp_operator[0] = 'l';
	(yyval.expression)->exp_operator[1] = 'e';
	(yyval.expression)->exp_operator[2] = 's';
	(yyval.expression)->exp_operator[3] = 's';
	(yyval.expression)->exp_operator[4] = 'e';
	(yyval.expression)->exp_operator[5] = 'r';
	(yyval.expression)->exp_operator[6] = 'e';
	(yyval.expression)->exp_operator[7] = 'q';
	(yyval.expression)->exp_operator[8] = 'u';
	(yyval.expression)->exp_operator[9] = 'a';
	(yyval.expression)->exp_operator[10] = 'l';
	if((yyvsp[-2].expression)->type_of_expr ==  1 || (yyvsp[0].expression)->type_of_expr == 1)
		YYABORT;
	(yyval.expression)->type_of_expr =  1;
}
#line 1806 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 41:
#line 359 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.expression),sizeof(ExpressionNode));
	cudaMallocHost((void**)(&((yyval.expression)->exp_operator)),12 * sizeof(char));
	(yyval.expression)->exp_operator[0] = 'd';
	(yyval.expression)->exp_operator[1] = 'o';
	(yyval.expression)->exp_operator[2] = 'u';
	(yyval.expression)->exp_operator[3] = 'b';
	(yyval.expression)->exp_operator[4] = 'l';
	(yyval.expression)->exp_operator[5] = 'e';
	(yyval.expression)->exp_operator[6] = 'e';
	(yyval.expression)->exp_operator[7] = 'q';
	(yyval.expression)->exp_operator[8] = 'u';
	(yyval.expression)->exp_operator[9] = 'a';
	(yyval.expression)->exp_operator[10] = 'l';
	(yyval.expression)->left_hand_term = (yyvsp[-2].expression);
	(yyval.expression)->right_hand_term = (yyvsp[0].expression);
	if((yyvsp[-2].expression)->type_of_expr ==  1 || (yyvsp[0].expression)->type_of_expr == 1)
		YYABORT;
	(yyval.expression)->type_of_expr =  1;
}
#line 1831 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 42:
#line 380 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.expression),sizeof(ExpressionNode));
	cudaMallocHost((void**)(&((yyval.expression)->exp_operator)),9 * sizeof(char));
	(yyval.expression)->left_hand_term = (yyvsp[-2].expression);
	(yyval.expression)->right_hand_term = (yyvsp[0].expression);
	(yyval.expression)->exp_operator[0] = 'n';
	(yyval.expression)->exp_operator[1] = 'o';
	(yyval.expression)->exp_operator[2] = 't';
	(yyval.expression)->exp_operator[3] = 'e';
	(yyval.expression)->exp_operator[4] = 'q';
	(yyval.expression)->exp_operator[5] = 'u';
	(yyval.expression)->exp_operator[6] = 'a';
	(yyval.expression)->exp_operator[7] = 'l';
	if((yyvsp[-2].expression)->type_of_expr ==  1 || (yyvsp[0].expression)->type_of_expr == 1)
		YYABORT;
	(yyval.expression)->type_of_expr =  1;
}
#line 1853 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 43:
#line 398 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.expression) = (yyvsp[0].expression);
}
#line 1861 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 44:
#line 403 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.expression),sizeof(ExpressionNode));
	cudaMallocHost((void**)(&((yyval.expression)->exp_operator)),5 * sizeof(char));
	(yyval.expression)->exp_operator[0] = 'p';
	(yyval.expression)->exp_operator[1] = 'l';
	(yyval.expression)->exp_operator[2] = 'u';
	(yyval.expression)->exp_operator[3] = 's';
	(yyval.expression)->left_hand_term = (yyvsp[-2].expression);
	(yyval.expression)->right_hand_term = (yyvsp[0].expression);
	if((yyvsp[-2].expression)->type_of_expr ==  1 || (yyvsp[0].expression)->type_of_expr == 1)
		YYABORT;
	(yyval.expression)->type_of_expr =  std::max((yyvsp[-2].expression)->type_of_expr,(yyvsp[0].expression)->type_of_expr);
}
#line 1879 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 45:
#line 417 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.expression),sizeof(ExpressionNode));
	cudaMallocHost((void**)(&((yyval.expression)->exp_operator)),6 * sizeof(char));
	(yyval.expression)->exp_operator[0] = 'm';
	(yyval.expression)->exp_operator[1] = 'i';
	(yyval.expression)->exp_operator[2] = 'n';
	(yyval.expression)->exp_operator[3] = 'u';
	(yyval.expression)->exp_operator[4] = 's';
	(yyval.expression)->left_hand_term = (yyvsp[-2].expression);
	(yyval.expression)->right_hand_term = (yyvsp[0].expression);
	if((yyvsp[-2].expression)->type_of_expr ==  1 || (yyvsp[0].expression)->type_of_expr == 1)
		YYABORT;
	(yyval.expression)->type_of_expr =  std::max((yyvsp[-2].expression)->type_of_expr,(yyvsp[0].expression)->type_of_expr);
}
#line 1898 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 46:
#line 432 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.expression) = (yyvsp[0].expression);
}
#line 1906 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 47:
#line 436 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.expression),sizeof(ExpressionNode));
	cudaMallocHost((void**)(&((yyval.expression)->exp_operator)),5 * sizeof(char));
	(yyval.expression)->exp_operator[0] = 'm';
	(yyval.expression)->exp_operator[1] = 'u';
	(yyval.expression)->exp_operator[2] = 'l';
	(yyval.expression)->exp_operator[3] = 't';
	(yyval.expression)->left_hand_term = (yyvsp[-2].expression);
	(yyval.expression)->right_hand_term = (yyvsp[0].expression);
	if((yyvsp[-2].expression)->type_of_expr ==  1 || (yyvsp[0].expression)->type_of_expr == 1)
		YYABORT;
	(yyval.expression)->type_of_expr =  std::max((yyvsp[-2].expression)->type_of_expr,(yyvsp[0].expression)->type_of_expr);
}
#line 1924 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 48:
#line 450 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.expression),sizeof(ExpressionNode));
	cudaMallocHost((void**)(&((yyval.expression)->exp_operator)),4 * sizeof(char));
	(yyval.expression)->exp_operator[0] = 'd';
	(yyval.expression)->exp_operator[1] = 'i';
	(yyval.expression)->exp_operator[2] = 'v';
	(yyval.expression)->left_hand_term = (yyvsp[-2].expression);
	(yyval.expression)->right_hand_term = (yyvsp[0].expression);
	if((yyvsp[-2].expression)->type_of_expr ==  1 || (yyvsp[0].expression)->type_of_expr == 1)
		YYABORT;
	(yyval.expression)->type_of_expr =  std::max((yyvsp[-2].expression)->type_of_expr,(yyvsp[0].expression)->type_of_expr);
}
#line 1941 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 49:
#line 463 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.expression),sizeof(ExpressionNode));
	cudaMallocHost((void**)(&((yyval.expression)->exp_operator)),7 * sizeof(char));
	(yyval.expression)->exp_operator[0] = 'm';
	(yyval.expression)->exp_operator[1] = 'o';
	(yyval.expression)->exp_operator[2] = 'd';
	(yyval.expression)->exp_operator[3] = 'u';
	(yyval.expression)->exp_operator[4] = 'l';
	(yyval.expression)->exp_operator[5] = 'o';
	(yyval.expression)->left_hand_term = (yyvsp[-2].expression);
	(yyval.expression)->right_hand_term = (yyvsp[0].expression);
	if((yyvsp[-2].expression)->type_of_expr !=  2 || (yyvsp[0].expression)->type_of_expr != 2)
		YYABORT;
	(yyval.expression)->type_of_expr =  2;
}
#line 1961 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 50:
#line 480 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.expression) = (yyvsp[0].expression);
}
#line 1969 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 51:
#line 485 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.expression),sizeof(ExpressionNode));
	cudaMallocHost((void**)(&((yyval.expression)->column_name)),sizeof(char)*(1+strlen((yyvsp[0].identifier))));
	strcpy((yyval.expression)->column_name,(yyvsp[0].identifier));
	(yyval.expression)->type_of_expr =  get_type((yyval.expression)->column_name);
}
#line 1980 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 52:
#line 492 "Expression.y" /* yacc.c:1646  */
    {
	cudaMallocHost((void**)&(yyval.expression),sizeof(ExpressionNode));
	(yyval.expression)->value = (yyvsp[0].value);
	(yyval.expression)->type_of_expr =  (floor((yyvsp[0].value)) == (yyvsp[0].value))?2:3;
}
#line 1990 "Expression.tab.cpp" /* yacc.c:1646  */
    break;

  case 53:
#line 498 "Expression.y" /* yacc.c:1646  */
    {
	(yyval.expression) = (yyvsp[-1].expression);
}
#line 1998 "Expression.tab.cpp" /* yacc.c:1646  */
    break;


#line 2002 "Expression.tab.cpp" /* yacc.c:1646  */
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYTERROR;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  return yyresult;
}
#line 501 "Expression.y" /* yacc.c:1906  */

void yyerror(const char* error_msg)
{
	std::cout<<error_msg<<'\n';
	return;
}

void update_query(SelectQuery* sq)
{
	select_query = sq;
}
void update_columns(std::vector<char*>* v)
{
	for(auto it: column_map)
	{
		const char* t = it.first.c_str();
		char* s;
		cudaMallocHost((void**)&s,strlen(t)+1);
		strcpy(s,t);
		v->push_back(s);
	}
}
int get_type(char* column)
{
	std::string s(column);
	return column_map[s];
}
bool find_column(char* column)
{
	std::string s(column);
	return (column_map.find(s) != column_map.end());
}

SelectQuery* process_query(std::string query)
{
	if(column_map.size() == 0)
	{
		column_map["vehicle_id"] = column_map["origin_vertex"] = column_map["destination_vertex"] = 2;
		column_map["database_index"] = 2;
		column_map["oil_life_pct"] = 3;
		column_map["tire_p_fl"] = column_map["tire_p_fr"] = column_map["tire_p_rl"] = column_map["tire_p_rr"] = 3;
		column_map["batt_volt"] = 3;
		column_map["fuel_percentage"] = 3;
		column_map["accel"] = 1;
		column_map["seatbelt"] = column_map["door_lock"] = column_map["hard_brake"] = column_map["gear_toggle"] = 1;
		column_map["clutch"] = column_map["hard_steer"] = 1;  
		column_map["speed"] = column_map["distance"] = 3;
	}
	select_query = NULL;
	yy_scan_string(query.c_str());
	yyparse();
	yylex_destroy();
	return select_query;
}
