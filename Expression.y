%{
	#include <iostream>
	#include <string>
	#include <stdlib.h>
	#include <vector>
	#include <map>
	#include <utility>
	#include <cmath>
	#include "proj_types.h"
	void yyerror(SelectQuery* select_query,std::map<std::string,int> column_map,const char* error_msg);
	//int yyparse(SelectQuery* select_query,std::map<std::string,int> column_map);
	int yylex(void);
	//int yylex_destroy(void);
	//int yy_scan_string(const char*);
	std::map<std::string,int> column_map;  
%}
%union
{
    double value;
    std::string* identifier;
	SelectQuery* SelectObject;
	bool distinct;
	ExpressionNode* expression;
	std::vector<std::string>* name_list;
	std::vector<std::pair<std::string,ExpressionNode*>>* expression_list;
	std::vector<std::pair<ExpressionNode*,bool>>* order_list;
}
%parse-param {SelectQuery* select_query}{std::map<std::string,int> column_map}
%start goal;
%type <SelectObject> goal Select_Query
%type <value> LimitExp Value
%type <distinct> DistinctQualifier
%type <expression_list> MultiAggCol AggCol
%type <name_list> SelectCol MultiCol
%type <order_list> OrderExp ExpList
%type <expression> Exp1 Exp2 Exp3 Exp Term WhereCondition GroupExp
%type <identifier> Column AggregateFunction Identifier OrderCriteria
%token Plus Minus Mult Div Modulo NotEqual Equal Greater GreaterEqual Lesser LesserEqual Or And Not Where Order Group By Limit Distinct Ascending Descending Comma OpeningBracket ClosingBracket Maximum Minimum Average Variance StandardDeviation Count Sum Identifier Value
%%
goal: Select_Query
{
	$$ = $1;
	select_query = $$;
	//std::cout<<"Reached Goal\n";
};
Select_Query: SelectCol DistinctQualifier WhereCondition GroupExp OrderExp LimitExp
{
	$$ = new SelectQuery();
	for(auto it: *$1)
	{
		if(column_map.find(it) == column_map.end())
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
	$$ = new SelectQuery();
	$$->aggregate_columns = *$1;
	$$->distinct_query = $2;
	$$->select_expression = $3;
	$$->group_term = $4;
	$$->order_term = *$5;
	$$->limit_term = $6;
	//std::cout<<"Reached Select_Query\n";
};
DistinctQualifier: Distinct
{
	$$ = true;
}
| %empty
{
	$$ = false;
};
Column: Identifier
{
	$$ = $1;	
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
WhereCondition: Where Exp
{
	$$ = $2;
	//std::cout<<"Reached WhereCond\n";
}
| %empty
{
	$$ = NULL;
	//std::cout<<"Reached WhereCond\n";
};
LimitExp: Limit Value
{
	$$ = $2;
}
| %empty
{
	$$ = -1;
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
AggCol: AggregateFunction OpeningBracket Exp ClosingBracket MultiAggCol
{
	$$ = $5;
	$$->push_back(std::make_pair($1,$3));
}
|  %empty
{
	$$ =  new std::vector<std::pair<std::string,ExpressionNode*>>;
};
MultiAggCol: MultiAggCol Comma AggregateFunction OpeningBracket Exp ClosingBracket
{
	$$ = $1;
	$$->push_back(std::make_pair(*$3,$5));
}
| %empty
{
	$$ = new std::vector<std::pair<std::string,ExpressionNode*>>;
};
SelectCol: Identifier MultiCol
{
	$$ = $2;
	$$->push_back(*$1);
	//std::cout<<"Reached Select_Col\n";
}
| Mult
{
	$$ = new std::vector<std::string>;
	for(auto it: column_map)
		$$->push_back(it.first);
	//std::cout<<"Reached Select_Col\n";	
};
MultiCol: MultiCol Comma Identifier
{
	$$ = $1;
	$1->push_back(*$3);
}	
|  %empty 
{
	$$ = new std::vector<std::string>;
};
GroupExp: Group By Exp
{
	$$ = $3;
}
| %empty
{
	$$ = NULL;
};
OrderExp: Order By Exp OrderCriteria ExpList
{
	$$ = $5;
	$$->push_back(std::make_pair($3,$4));
}
| Order By Exp ExpList
{
	$$ = $4;
	$$->push_back(std::make_pair($3,true));
}
| %empty
{
	$$ = new std::vector<std::pair<ExpressionNode*,bool>>;
};
ExpList: ExpList Comma Exp
{
	$$ = $1;
	$$.push_back(std::make_pair($3,true));
}
| ExpList Comma Exp OrderCriteria 
{
	$$ = $1;
	$$->push_back(std::make_pair($3,$4));
}
| %empty
{
	$$ = new std::vector<std::pair<ExpressionNode*,bool>>;
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
	$$->exp_operator = "Not";
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
	$$->column_name = $1;
	$$->type_of_expr =  column_map[$1];
}
| Value
{
	$$ = new ExpressionNode();
	$$->value = $1;
	$$->type_of_expr =  (floor($1) == $1)?2:3;
}
| OpeningBracket Exp ClosingBracket
{
	$$ = $2;
};
%%
void yyerror(SelectQuery* select_query,std::map<std::string,int> column_map,const char* error_msg)
{
	std::cout<<"Failed due to: "<<error_msg<<'\n';
	return;
}
int main(void)
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
	SelectQuery* select_query;
	yyparse(select_query,column_map);
	std::cout<<select_query<<'\n';
}
//SelectQuery* process_query(std::string query)
//{
	//if(column_map.size() == 0)
	//{
		//column_map["vehicle_id"] = 2;
		//column_map["database_index"] = 2;
		//column_map["oil_life_pct"] = 3;
		//column_map["tire_p_fl"] = column_map["tire_p_fr"] = column_map["tire_p_rl"] = column_map["tire_p_rr"] = 3;
		//column_map["batt_volt"] = 3;
		//column_map["fuel_percentage"] = 3;
		//column_map["accel"] = 1;
		//column_map["seatbelt"] = column_map["door_lock"] = column_map["hard_brake"] = column_map["gear_toggle"] = 1;
		//column_map["clutch"] = column_map["hard_steer"] = 1;  
		//column_map["speed"] = column_map["distance"] = 3;
//	}
	//SelectQuery* select_query;
	//yy_scan_string(query.c_str());
	//yyparse(select_query,column_map);
	//yylex_destroy();
	//return select_query;
//}
