%{
	#include <iostream>
	#include <string>
	#include <string.h>
	#include <stdlib.h>
	#include <utility>
	#include <vector>
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
%}
%union
{
	double value;
	char* identifier;
	class SelectQuery* SelectObject;
	bool distinct;
	class ExpressionNode* expression;
	std::vector<char*>* name_list;
	std::vector<std::pair<char*,ExpressionNode*>>* expression_list;
	std::vector<std::pair<ExpressionNode*,bool>>* order_list;
}
%start goal;
%type <SelectObject> Select_Query goal
%type <value> LimitExp 
%type <distinct> OrderCriteria
%type <expression_list> MultiAggCol AggCol
%type <name_list> Columns MultiCol
%type <order_list> OrderExp ExpList
%type <expression> Exp1 Exp2 Exp3 Exp Term WhereCondition GroupExp 
%type <identifier> AggregateFunction
%token Plus Minus Mult Div Modulo NotEqual DoubleEqual Greater GreaterEqual Lesser LesserEqual Or And Not Select Where Order Group By Limit Distinct Ascending Descending Comma OpeningBracket ClosingBracket Maximum Minimum Average Variance StandardDeviation Count Sum Identifier Value
%%
goal: Select_Query
{
	$$ = $1;
	update_query($$);
};
Select_Query: Select Columns WhereCondition GroupExp OrderExp LimitExp
{
	$$ = new SelectQuery;
	for(auto it: *$2)
	{
		if(!find_column(it))
			YYABORT;
	}
	$$->select_columns = *$2;
	$$->select_expression = $3;
	$$->group_term = $4;
	$$->order_term = *$5;
	$$->limit_term = $6;
	//std::cout<<"Reached Select_Query\n";
}
| Select AggCol WhereCondition GroupExp OrderExp LimitExp
{
	$$ = new SelectQuery;
	$$->select_expression =  $3;
	$$->group_term = $4;
	$$->order_term = *$5;
	$$->limit_term = $6;  
};

Columns: Identifier MultiCol
{
	$$ = $2;
	$$->push_back(yylval.identifier);
}
|
Mult
{
	$$ = new std::vector<char*>();
	update_columns($$);
};
MultiCol: MultiCol Comma Identifier
{
	$$ = $1;
	$$->push_back(yylval.identifier);
}
| 
/*empty*/
{
	$$ = new std::vector<char*>();
};
OrderCriteria: Ascending
{
	$$ = true;
	//std::cout<<"Reached Order:Asc\n";
}
| Descending
{
	$$ = false;
	//std::cout<<"Reached Order:desc\n";
}
|
/*empty*/
{
	$$ = true;
};
WhereCondition: 
 Where Exp
{
	$$ = $2;
	//std::cout<<"Reached WhereCond\n";
}
|
/*empty*/
{
	$$ = NULL;
};
LimitExp: 
Limit Value
{
	$$ = floor(yylval.value);
}
|
/*empty*/
{
	$$ = -1;
};

AggregateFunction: Maximum
{
	$$ = "max";
}
| Minimum
{
	$$ = "min";
}
| Average
{
	$$ = "avg";
}
| Variance
{
	$$ = "var";
}
| StandardDeviation
{
	$$ = "std";
}
| Count
{
	$$ = "count";
}
| Sum
{
	$$ = "sum";
};
AggCol: 
AggregateFunction OpeningBracket Exp ClosingBracket MultiAggCol
{
	$$ = $5;
	$$->push_back(std::make_pair($1,$3));
};
MultiAggCol: Comma AggregateFunction OpeningBracket Exp ClosingBracket MultiAggCol
{
	$$ = $6;
	$$->push_back(std::make_pair($2,$4));
}
|
/*empty*/
{
	$$ = new std::vector<std::pair<char*,ExpressionNode*>>;
};
GroupExp: 
Group By Exp
{
	$$ = $3;
}
|
/*empty*/
{
	$$ = NULL;
};

OrderExp: 
Order By ExpList
{
	$$ = $3;
}
|
/*empty*/
{
	$$ = NULL;
};
ExpList: 
Exp OrderCriteria ExpList
{
	$$ = $3;
	$$->push_back(std::make_pair($1,$2));
}
| 
Exp OrderCriteria 
{
	$$ = new std::vector<std::pair<ExpressionNode*,bool>>();
	$$->push_back(std::make_pair($1,$2));
}; 

Exp: Exp Or Exp1
{
	$$ = new ExpressionNode;
	$$->exp_operator = "or";
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	$$->type_of_expr =  1;
}
| Exp And Exp1
{
	$$ = new ExpressionNode;
	$$->exp_operator = "and";
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	$$->type_of_expr =  1;
}
| Not Exp1
{
	$$ = new ExpressionNode;
	$$->exp_operator = "not";
	$$->left_hand_term = $2;
	if($$->type_of_expr != 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp1
{
	$$=$1;
};

Exp1: Exp1 Greater Exp2
{
	$$ = new ExpressionNode;
	$$->exp_operator = "greater";
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp1 Lesser Exp2
{
	$$ = new ExpressionNode;
	$$->exp_operator = "lesser";
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp1 GreaterEqual Exp2
{
	$$ = new ExpressionNode;
	$$->exp_operator = "greaterequal";
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp1 LesserEqual Exp2
{
	$$ = new ExpressionNode;
	$$->exp_operator = "lesserequal";
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp1 DoubleEqual Exp2
{
	$$ = new ExpressionNode;
	$$->exp_operator = "doubleequal";
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp1 NotEqual Exp2
{
	$$ = new ExpressionNode;
	$$->exp_operator = "notequal";
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp2
{
	$$ = $1;
};

Exp2: Exp2 Plus Exp3
{
	$$ = new ExpressionNode;
	$$->exp_operator = "plus";
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  std::max($1->type_of_expr,$3->type_of_expr);
}
| Exp2 Minus Exp3
{
	$$ = new ExpressionNode;
	$$->exp_operator = "minus";
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  std::max($1->type_of_expr,$3->type_of_expr);
}
| Exp3
{
	$$ = $1;
};
Exp3: Exp3 Mult Term
{
	$$ = new ExpressionNode;
	$$->exp_operator = "mult";
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  3;
}
| Exp3 Div Term
{
	$$ = new ExpressionNode;
	$$->exp_operator = "div";
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  3;
}
| Exp3 Modulo Term
{
	$$ = new ExpressionNode;
	$$->exp_operator = "modulo";
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr !=  2 || $3->type_of_expr != 2)
		YYABORT;
	$$->type_of_expr =  2;
}
|
Term
{
	$$ = $1;
};

Term: Identifier
{
	$$ = new ExpressionNode;
	$$->column_name = yylval.identifier;
	$$->type_of_expr =  get_type($$->column_name);
}
| Value
{
	$$ = new ExpressionNode;
	$$->value = yylval.value;
	$$->type_of_expr =  (floor(yylval.value) == yylval.value)?2:3;
}
| OpeningBracket Exp ClosingBracket
{
	$$ = $2;
};
%%
void yyerror(const char* error_msg)
{
	std::cout<<"Failed due to: "<<error_msg<<'\n';
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
		char* s = new char[strlen(t)+1];
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
		column_map["vehicle_id"] = 2;
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
