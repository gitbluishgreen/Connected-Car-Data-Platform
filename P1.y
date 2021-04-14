%{
	#include <stdio.h>
	#include <string.h>
	#include <stdlib.h>
	void yyerror (const char* h);
	int yylex(void);
	int yywrap(void);
	struct node
	{
		struct node* left;
		struct node* right;
		char op;
	};
	struct query
	{
		struct node* where_exp;
		struct node* group_exp;
		struct node* order_exp;
		int limit;
		bool distinct;
	};
%}
%union
{
    int integer;
    char* string;
    struct node* expr_node;
    struct query* query;
    char** array;
}
%start goal;
%type <query> SelectQuery
%type <array> Columns DistinctExp WhereExp GroupExp OrderExp LimitExp AggCol MultiAggCol MultiColumns SelectCols AggValue
%type <expr_node> Expression Expression1 Expression2 Expression3 Expression4
%token Plus Minus Mult Div Modulo Identifier OpeningBracket ClosingBracket GreaterThan LessThan LessEqual GreaterEqual DoubleEqual NotEqual Or And Not Integer Select Where Order Group Limit Distinct Comma By Min Max Average Variance StdDev Count Sum
%%
goal: SelectQuery
{
};
SelectQuery: Select DistinctExp SelectCols WhereExp GroupExp OrderExp LimitExp
{
};
DistinctExp: Distinct
{
}
|
/*empty*/
{
};
MultiColumns: MultiColumns Comma Identifier
{
}
|
/*empty*/
{
};
Columns: Identifier MultiColumns
{
}
| Mult
{
};
AggCol: AggValue OpeningBracket Expression ClosingBracket MultiAggCol
{
};
MultiAggCol: Comma AggValue OpeningBracket Expression ClosingBracket MultiAggCol
{
}
| 
/*empty*/
{
};
AggValue: Max {}
| Min {}
| Average {}
| StdDev {}
| Variance {}
| Sum {}
| Count {};

SelectCols: Columns
{
}
|
AggCol
{
};
WhereExp: Where Expression
{
}
| 
/*empty*/
{
};
GroupExp: Group By Expression
{
}
|
/*empty*/
{
};
OrderExp: Order By Expression
{
}
|
/*empty*/
{
};
LimitExp: Limit Integer
{
}
|
/*empty*/
{
};
Expression: Expression And Expression1
{
}
| Expression Or Expression1
{
}
| Not Expression1
{
}
| Expression1
{
};

Expression1: Expression1 DoubleEqual Expression2
{
}
| Expression1 NotEqual Expression2
{
}
| Expression1 GreaterThan Expression2
{
}
| Expression1 LessThan Expression2
{
}
| Expression1 GreaterEqual Expression2
{
}
| Expression1 LessEqual Expression2
{
}
| Expression2
{
};
Expression2: Expression2 Plus Expression3
{
}
| Expression2 Minus Expression3
{
}
| Expression3
{
};
Expression3: Expression3 Mult Expression4
{
}
| Expression3 Div Expression4
{
}
| Expression3 Modulo Expression4
{
}
| Expression4 
{
};
Expression4: Identifier
{
}
| Integer
{
}
| OpeningBracket Expression ClosingBracket
{
};
%%

void yyerror(const char *s)
{
	printf ("%s\n",s);
	exit(0);
}

int main ()
{
	if(yyparse() == 0);
		printf("Successfully parsed!\n");
	return 0;
}
