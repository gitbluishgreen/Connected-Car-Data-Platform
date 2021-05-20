%{
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
%type <value> LimitExp Value
%type <distinct> OrderCriteria DistinctQualifier
%type <expression_list> MultiAggCol AggCol
%type <name_list> Columns MultiCol
%type <order_list> OrderExp ExpList
%type <expression> Exp1 Exp2 Exp3 Exp Term WhereCondition GroupExp 
%type <identifier> AggregateFunction Identifier
%token Plus Minus Mult Div Modulo NotEqual DoubleEqual Greater GreaterEqual Lesser LesserEqual Or And Not Select Distinct Where Order Group By Limit Ascending Descending Comma OpeningBracket ClosingBracket Maximum Minimum Average Variance StandardDeviation Sum Identifier Value
%%
goal: Select_Query
{
	$$ = $1;
	update_query($$);
};
Select_Query: Select DistinctQualifier Columns WhereCondition OrderExp LimitExp
{
	for(auto it: *$3)
	{
		if(!find_column(it))
			YYABORT;
	}
	cudaMallocHost((void**)&$$,sizeof(SelectQuery));//pinned memory
	$$->distinct = $2;
	std::reverse($3->begin(),$3->end());
	$$->select_columns = $3;
	$$->select_expression = $4;
	$$->order_term = $5;
	$$->limit_term = $6;
}
| Select AggCol WhereCondition GroupExp OrderExp LimitExp
{
	cudaMallocHost((void**)&$$,sizeof(SelectQuery));
	std::reverse($2->begin(),$2->end());
	$$->distinct = false;
	$$->aggregate_columns = $2;
	$$->select_expression =  $3;
	$$->group_term = $4;
	$$->order_term = $5;
	$$->limit_term = $6;  
};

DistinctQualifier: Distinct
{
	$$ = true;
}
| 

/*empty*/
{
	$$ = false;
};

Columns: Identifier MultiCol
{
	$$ = $2;
	$$->push_back($1);
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
	$$->push_back($3);
}
| 
/*empty*/
{
	$$ = new std::vector<char*>();
};
OrderCriteria: Ascending
{
	$$ = true;
}
| Descending
{
	$$ = false;
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
}
|
/*empty*/
{
	$$ = NULL;
};
LimitExp: 
Limit Value
{
	$$ = floor($2);
}
|
/*empty*/
{
	$$ = -1;
};

AggregateFunction: Maximum
{
	cudaMallocHost((void**)&$$,4*sizeof(char));
	$$[0] = 'm';
	$$[1] = 'a';
	$$[2] = 'x';
}
| Minimum
{
	cudaMallocHost((void**)&$$,4*sizeof(char));
	$$[0] = 'm';
	$$[1] = 'i';
	$$[2] = 'n';
}
| Average
{
	cudaMallocHost((void**)&$$,4*sizeof(char));
	$$[0] = 'a';
	$$[1] = 'v';
	$$[2] = 'g';
}
| Variance
{
	cudaMallocHost((void**)&$$,4*sizeof(char));
	$$[0] = 'v';
	$$[1] = 'a';
	$$[2] = 'r';
}
| StandardDeviation
{
	cudaMallocHost((void**)&$$,4*sizeof(char));
	$$[0] = 's';
	$$[1] = 't';
	$$[2] = 'd';
}
| Sum
{
	cudaMallocHost((void**)&$$,4*sizeof(char));
	$$[0] = 's';
	$$[1] = 'u';
	$$[2] = 'm';
};
AggCol: 
AggregateFunction OpeningBracket Exp ClosingBracket MultiAggCol
{
	$$ = $5;
	if($3->type_of_expr == 1)
		YYABORT;//no boolean values allowed.
	$$->push_back(std::make_pair($1,$3));
};
MultiAggCol: MultiAggCol Comma AggregateFunction OpeningBracket Exp ClosingBracket
{
	$$ = $1;
	if($5->type_of_expr == 1)
		YYABORT;//no boolean value allowed.
	$$->push_back(std::make_pair($3,$5));
}
|
/*empty*/
{
	$$ = new std::vector<std::pair<char*,ExpressionNode*>>;//do we need to replace by thrust?
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
ExpList Exp OrderCriteria
{
	$$ = $1;
	$$->push_back(std::make_pair($2,$3));
}
| 
Exp OrderCriteria 
{
	$$ = new std::vector<std::pair<ExpressionNode*,bool>>();//replace by thrust?
	$$->push_back(std::make_pair($1,$2));
}; 

Exp: Exp Or Exp1
{
	cudaMallocHost((void**)&$$,sizeof(ExpressionNode));
	cudaMallocHost((void**)(&($$->exp_operator)),3 * sizeof(char));
	$$->exp_operator[0] = 'o';
	$$->exp_operator[1] = 'r';
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr != 1 || $3->type_of_expr!= 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp And Exp1
{
	cudaMallocHost((void**)&$$,sizeof(ExpressionNode));
	cudaMallocHost((void**)(&($$->exp_operator)),4 * sizeof(char));
	$$->exp_operator[0] = 'a';
	$$->exp_operator[1] = 'n';
	$$->exp_operator[2] = 'd';
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr != 1 || $3->type_of_expr!= 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Not Exp1
{
	cudaMallocHost((void**)&$$,sizeof(ExpressionNode));
	cudaMallocHost((void**)(&($$->exp_operator)),4 * sizeof(char));
	$$->exp_operator[0] = 'n';
	$$->exp_operator[1] = 'o';
	$$->exp_operator[2] = 't';
	$$->left_hand_term = $2;
	if($2->type_of_expr != 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp1
{
	$$=$1;
};

Exp1: Exp1 Greater Exp2
{
	cudaMallocHost((void**)&$$,sizeof(ExpressionNode));
	cudaMallocHost((void**)(&($$->exp_operator)),8 * sizeof(char));
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	$$->exp_operator[0] = 'g';
	$$->exp_operator[1] = 'r';
	$$->exp_operator[2] = 'e';
	$$->exp_operator[3] = 'a';
	$$->exp_operator[4] = 't';
	$$->exp_operator[5] = 'e';
	$$->exp_operator[6] = 'r';
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp1 Lesser Exp2
{
	cudaMallocHost((void**)&$$,sizeof(ExpressionNode));
	cudaMallocHost((void**)(&($$->exp_operator)),7 * sizeof(char));
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	$$->exp_operator[0] = 'l';
	$$->exp_operator[1] = 'e';
	$$->exp_operator[2] = 's';
	$$->exp_operator[3] = 's';
	$$->exp_operator[4] = 'e';
	$$->exp_operator[5] = 'r';
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp1 GreaterEqual Exp2
{
	cudaMallocHost((void**)&$$,sizeof(ExpressionNode));
	cudaMallocHost((void**)(&($$->exp_operator)),13 * sizeof(char));
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	$$->exp_operator[0] = 'g';
	$$->exp_operator[1] = 'r';
	$$->exp_operator[2] = 'e';
	$$->exp_operator[3] = 'a';
	$$->exp_operator[4] = 't';
	$$->exp_operator[5] = 'e';
	$$->exp_operator[6] = 'r';
	$$->exp_operator[7] = 'e';
	$$->exp_operator[8] = 'q';
	$$->exp_operator[9] = 'u';
	$$->exp_operator[10] = 'a';
	$$->exp_operator[11] = 'l';
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp1 LesserEqual Exp2
{
	cudaMallocHost((void**)&$$,sizeof(ExpressionNode));
	cudaMallocHost((void**)(&($$->exp_operator)),12 * sizeof(char));
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	$$->exp_operator[0] = 'l';
	$$->exp_operator[1] = 'e';
	$$->exp_operator[2] = 's';
	$$->exp_operator[3] = 's';
	$$->exp_operator[4] = 'e';
	$$->exp_operator[5] = 'r';
	$$->exp_operator[6] = 'e';
	$$->exp_operator[7] = 'q';
	$$->exp_operator[8] = 'u';
	$$->exp_operator[9] = 'a';
	$$->exp_operator[10] = 'l';
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp1 DoubleEqual Exp2
{
	cudaMallocHost((void**)&$$,sizeof(ExpressionNode));
	cudaMallocHost((void**)(&($$->exp_operator)),12 * sizeof(char));
	$$->exp_operator[0] = 'd';
	$$->exp_operator[1] = 'o';
	$$->exp_operator[2] = 'u';
	$$->exp_operator[3] = 'b';
	$$->exp_operator[4] = 'l';
	$$->exp_operator[5] = 'e';
	$$->exp_operator[6] = 'e';
	$$->exp_operator[7] = 'q';
	$$->exp_operator[8] = 'u';
	$$->exp_operator[9] = 'a';
	$$->exp_operator[10] = 'l';
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  1;
}
| Exp1 NotEqual Exp2
{
	cudaMallocHost((void**)&$$,sizeof(ExpressionNode));
	cudaMallocHost((void**)(&($$->exp_operator)),9 * sizeof(char));
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	$$->exp_operator[0] = 'n';
	$$->exp_operator[1] = 'o';
	$$->exp_operator[2] = 't';
	$$->exp_operator[3] = 'e';
	$$->exp_operator[4] = 'q';
	$$->exp_operator[5] = 'u';
	$$->exp_operator[6] = 'a';
	$$->exp_operator[7] = 'l';
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
	cudaMallocHost((void**)&$$,sizeof(ExpressionNode));
	cudaMallocHost((void**)(&($$->exp_operator)),5 * sizeof(char));
	$$->exp_operator[0] = 'p';
	$$->exp_operator[1] = 'l';
	$$->exp_operator[2] = 'u';
	$$->exp_operator[3] = 's';
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  std::max($1->type_of_expr,$3->type_of_expr);
}
| Exp2 Minus Exp3
{
	cudaMallocHost((void**)&$$,sizeof(ExpressionNode));
	cudaMallocHost((void**)(&($$->exp_operator)),6 * sizeof(char));
	$$->exp_operator[0] = 'm';
	$$->exp_operator[1] = 'i';
	$$->exp_operator[2] = 'n';
	$$->exp_operator[3] = 'u';
	$$->exp_operator[4] = 's';
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
	cudaMallocHost((void**)&$$,sizeof(ExpressionNode));
	cudaMallocHost((void**)(&($$->exp_operator)),5 * sizeof(char));
	$$->exp_operator[0] = 'm';
	$$->exp_operator[1] = 'u';
	$$->exp_operator[2] = 'l';
	$$->exp_operator[3] = 't';
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  3;
}
| Exp3 Div Term
{
	cudaMallocHost((void**)&$$,sizeof(ExpressionNode));
	cudaMallocHost((void**)(&($$->exp_operator)),4 * sizeof(char));
	$$->exp_operator[0] = 'd';
	$$->exp_operator[1] = 'i';
	$$->exp_operator[2] = 'v';
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type_of_expr ==  1 || $3->type_of_expr == 1)
		YYABORT;
	$$->type_of_expr =  3;
}
| Exp3 Modulo Term
{
	cudaMallocHost((void**)&$$,sizeof(ExpressionNode));
	cudaMallocHost((void**)(&($$->exp_operator)),7 * sizeof(char));
	$$->exp_operator[0] = 'm';
	$$->exp_operator[1] = 'o';
	$$->exp_operator[2] = 'd';
	$$->exp_operator[3] = 'u';
	$$->exp_operator[4] = 'l';
	$$->exp_operator[5] = 'o';
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
	cudaMallocHost((void**)&$$,sizeof(ExpressionNode));
	cudaMallocHost((void**)(&($$->column_name)),sizeof(char)*(1+strlen($1)));
	strcpy($$->column_name,$1);
	$$->type_of_expr =  get_type($$->column_name);
}
| Value
{
	cudaMallocHost((void**)&$$,sizeof(ExpressionNode));
	$$->value = $1;
	$$->type_of_expr =  (floor($1) == $1)?2:3;
}
| OpeningBracket Exp ClosingBracket
{
	$$ = $2;
};
%%
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