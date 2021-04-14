%{
	#include <iostream>
	#include <string>
	#include <stdlib.h>
	#include <utility>
	#include <vector>
	#include <map>
	#include "proj_types.hpp"
	void yyerror(const char*);
	int yyparse(void);
	int yylex(void);
	int yy_scan_string(const char*);
	int yylex_destroy(void);
	void update_query(SelectQuery*);
	int get_type(std::string);
	void update_columns(std::vector<std::string>*);
	bool find_column(std::string);
	std::map<std::string,int> column_map; 
	SelectQuery* select_query;
%}
%union
{
    double value;
    std::string* identifier;
	class SelectQuery* SelectObject;
	bool distinct;
	class ExpressionNode* expression;
	std::vector<std::string>* name_list;
	std::vector<std::pair<std::string,ExpressionNode*>>* expression_list;
	std::vector<std::pair<ExpressionNode*,bool>>* order_list;
}
%start goal;
%type <SelectObject> Select_Query goal
%type <value> LimitExp 
%type <distinct> DistinctQualifier
%type <expression_list> MultiAggCol AggCol
%type <name_list> SelectCol MultiCol
%type <order_list> OrderExp ExpList
%type <expression> Exp1 Exp2 Exp3 Exp Term WhereCondition GroupExp 
%type <identifier> Column AggregateFunction OrderCriteria
%token Plus Minus Mult Div Modulo NotEqual Equal Greater GreaterEqual Lesser LesserEqual Or And Not Where Order Group By Limit Distinct Ascending Descending Comma OpeningBracket ClosingBracket Maximum Minimum Average Variance StandardDeviation Count Sum Identifier Value
%%
goal: Select_Query
{
	$$ = $1;
	//update_query($$);
};
Select_Query: SelectCol DistinctQualifier WhereCondition GroupExp OrderExp LimitExp
{
	$$ = new SelectQuery;
	$$->distinct_query = false;
	for(auto it: *$1)
	{
		if(!find_column(it))
			YYABORT;
	}
	$$->select_columns = *$1;
	$$->distinct_query = $2;
	$$->select_expression = $3;
	$$->group_term = $4;
	$$->order_term = *$5;
	$$->limit_term = $6;
	//std::cout<<"Reached Select_Query\n";
}
| AggCol DistinctQualifier WhereCondition GroupExp OrderExp LimitExp
{
	$$ = new SelectQuery;
	$$->distinct_query = false;
	$$->aggregate_columns = *$1;
	$$->distinct_query = $2;
	$$->select_expression = $3;
	$$->group_term = $4;
	$$->order_term = *$5;
	$$->limit_term = $6;
	//std::cout<<"Reached Select_Query\n";
};
DistinctQualifier: 
Mult
{
	$$ = false;
}
| Distinct
{
	$$ = true;
};
Column: Identifier
{
	$$ = *(yylval.identifier);	
};
OrderCriteria: Ascending
{
	*$$ = "asc";
	//std::cout<<"Reached Order:Asc\n";
}
| Descending
{
	*$$ = "desc";
	//std::cout<<"Reached Order:desc\n";
};
WhereCondition: 
 Where Mult
{
	$$ = NULL;
	//std::cout<<"Reached WhereCond\n";
}
| Where Exp
{
	$$ = $2;
	//std::cout<<"Reached WhereCond\n";
};
LimitExp: 
LIMIT Mult
{
	$$ = -1;
} | Limit Value
{
	$$ = yylval.value;
};

AggregateFunction: Maximum
{
	$$ = new std::string("max");
}
| Minimum
{
	$$ = new std::string("min");
}
| Average
{
	$$ = new std::string("avg");
}
| Variance
{
	$$ = new std::string("var");
}
| StandardDeviation
{
	$$ = new std::string("std");
}
| Count
{
	$$ = new std::string("count");
}
| Sum
{
	$$ = new std::string("sum");
};
AggCol: 
 Mult
{
	$$ =  new std::vector<std::pair<std::string,ExpressionNode*>>;
}
| AggregateFunction OpeningBracket Exp ClosingBracket MultiAggCol
{
	$$ = $5;
	$$->push_back(std::make_pair($1,$3));
};
MultiAggCol: 
/*empty*/
{
	$$ = new std::vector<std::pair<std::string,ExpressionNode*>>;
}
| MultiAggCol Comma AggregateFunction OpeningBracket Exp ClosingBracket
{
	$$ = $1;
	$$->push_back(std::make_pair(*$3,$5));
};
SelectCol: Identifier MultiCol
{
	$$ = $2;
	$$->push_back(*(yylval.identifier));
	//std::cout<<"Reached Select_Col\n";
}
| Mult
{
	$$ = new std::vector<std::string>;
	update_columns($$);
	//std::cout<<"Reached Select_Col\n";	
};

MultiCol:
/*empty*/
{
	$$ = new std::vector<std::string>;
}
|  MultiCol Comma Identifier
{
	$$ = $1;
	$1->push_back(*(yylval.identifier));
};

GroupExp: 
Group By Mult
{
	$$ = NULL;
}
| Group By Exp
{
	$$ = $3;
};

OrderExp: 
Order By Mult
{
	$$ = new std::vector<std::pair<ExpressionNode*,bool>>;
}
| Order By Exp OrderCriteria ExpList
{
	$$ = $5;
	$$->push_back(std::make_pair($3,$4));
}
| Order By Exp ExpList
{
	$$ = $4;
	$$->push_back(std::make_pair($3,true));
};

ExpList: 
/*empty*/
{
	$$ = new std::vector<std::pair<ExpressionNode*,bool>>;
}
| ExpList Comma Exp
{
	$$ = $1;
	$$.push_back(std::make_pair($3,true));
}
| ExpList Comma Exp OrderCriteria 
{
	$$ = $1;
	$$->push_back(std::make_pair($3,$4));
}; 

Exp: Exp Or Exp1
{
	$$ = new ExpressionNode("or");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	$$->type_of_expr =  1;
}
| Exp And Exp1
{
	$$ = new ExpressionNode("and");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	$$->type_of_expr =  1;
}
| Not Exp1
{
	$$ = new ExpressionNode();
	$$->exp_operator = "not";
	$$->left_hand_term = $2;
	if($$->type != 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp1
{
	$$=$1;
};

Exp1: Exp1 Greater Exp2
{
	$$ = new ExpressionNode("greater");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp1 Lesser Exp2
{
	$$ = new ExpressionNode("greater");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp1 GreaterEqual Exp2
{
	$$ = new ExpressionNode("GreaterEqual");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp1 LesserEqual Exp2
{
	$$ = new ExpressionNode("LesserEqual");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp1 Equal Exp2
{
	$$ = new ExpressionNode("Equal");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp1 NotEqual Exp2
{
	$$ = new ExpressionNode("NotEqual");
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
	$$ = new ExpressionNode("Plus");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  max($1->type,$3->type);
}
| Exp2 Minus Exp3
{
	$$ = new ExpressionNode("NotEqual");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  max($1->type,$3->type);
}
| Exp3
{
	$$ = $1;
};
Exp3: Exp3 Mult Term
{
	$$ = new ExpressionNode("Mult");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  3;
}
| Exp3 Div Term
{
	$$ = new ExpressionNode("Div");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  3;
}
| Exp3 Modulo Term
{
	$$ = new ExpressionNode("NotEqual");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type !=  2 || $3->type != 2)
		YYABORT;
	$$->type_of_expr =  2;
};
Term: Column
{
	$$ = new ExpressionNode();
	$$->column_name = *(yylval.identfier);
	$$->type_of_expr =  get_type($$->column_name);
}
| Value
{
	$$ = new ExpressionNode();
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
void update_columns(std::vector<std::string>* v)
{
	for(auto it: column_map)
	{
		v->push_back(it.first);
	}
}
int get_type(std::string column)
{
	return column_map[column];
}
bool find_column(std::string column)
{
	return (column_map.find(column) != column_map.end());
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
