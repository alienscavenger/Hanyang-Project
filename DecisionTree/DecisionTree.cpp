#include <cstdio>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include <map>
#include <utility>
#include <queue>

using namespace std;
typedef pair<double, string> pds;

string training_name;
string test_name;
string output_name;
ofstream out;
string classifier_name; // to make finding the final label easier
bool DEBUG = false;

class Classifier
{
public:
	string name;
	map<string, int> label;
	Classifier(string name) : name(name) {}
};

class Attribute
{
public:
	string name;
	map<string, int> label;
	map<pair<string, string>, int> class_tuple; // < <label, class_label>, occurence>
	Attribute(string name) : name(name) {};
};

class DT_node
{
public:
	string name;
	string class_label;
	vector<pair<string, DT_node*> > childs; // pair<condition, next_node>
	DT_node()
	{
		class_label = "<non_leaf>";
	}
};

vector<vector<string> > database;
vector<vector<string> > test_dataset;
queue < pair< vector<vector<string> >*, DT_node*> > DT_build_queue;

void scanFile()
{
	ifstream inputFile(training_name);
	string line;
	while (getline(inputFile, line))
	{
		vector<string> row; row.clear();
		string item; item.clear();
		stringstream ss = stringstream(line);
		while (ss >> item)
		{
			row.push_back(item);
		}
		database.push_back(row);
	}
	classifier_name = database[0][database[0].size() - 1];
}

void read_dataset(vector<vector<string> >* dataset, Classifier* classifier, vector<Attribute>* attributes, int row_sz, int col_sz)
{
	for (int row = 1; row < row_sz; row++) // start from the first data, skipping the header
	{
		string classifier_label = dataset->at(row)[col_sz - 1];
		classifier->label[classifier_label]++; // increase the count for the classifier's label
		for (int col = 0; col < col_sz - 1; col++) // for all attributes EXCEPT the classifier
		{
			string attribute_label = dataset->at(row)[col];
			attributes->at(col).label[attribute_label]++;
			attributes->at(col).class_tuple[make_pair(attribute_label, classifier_label)]++;
		}
	}
}

void DEBUG_attribute_count(vector<Attribute>* attributes)
{
	vector<Attribute>::iterator it = attributes->begin();
	while (it != attributes->end())
	{
		map<string, int>::iterator itt = it->label.begin();
		cout << it->name << ":\n";
		while (itt != it->label.end())
		{
			cout << "(" + itt->first + "," + to_string(itt->second) + ")" << '\t';
			itt++;
		}
		cout << "\n";
		it++;
	}
	cout << "---\n";

	it = attributes->begin();
	while (it != attributes->end())
	{
		map<pair<string, string>, int>::iterator itt = it->class_tuple.begin();
		cout << it->name << ":\n";
		while (itt != it->class_tuple.end())
		{
			cout << "(" + itt->first.first + "," + itt->first.second + ")," + to_string(itt->second) + '\t' << endl;
			itt++;
		}
		cout << "\n";

		it++;
	}
	cout << "\n";
}

void get_info_parent(Classifier* classifier, int total_data, double& info_parent)
{
	map<string, int>::iterator classifier_label_iter = classifier->label.begin();
	while (classifier_label_iter != classifier->label.end())
	{
		double count = (double)classifier_label_iter->second;
		info_parent += -(count / total_data)*log2((count / total_data));
		classifier_label_iter++;
	}
}

class Custom_Comparator
{
public:
	bool operator()(pds a, pds b)
	{
		return a.first > b.first; //ascending (the lower the Info.x, the higher the Information Gain)
	}
};

bool all_same_class(vector<vector<string> >* dataset)
{
	int col_sz = dataset->at(0).size();
	int row_sz = dataset->size();
	string comp = dataset->at(1).at(col_sz - 1);
	for (int i = 2; i < row_sz; i++)
	{
		if (dataset->at(i).at(col_sz - 1) != comp) return false;
	}
	return true;
}

void get_maximum_gain(vector<Attribute>* attributes, int col_sz, int total_data, double info_parent, double& maximum,
	int& selected_att_idx, vector < priority_queue<pds, vector<pds>, Custom_Comparator> >* pq)
{
	for (int col = 0; col < col_sz - 1; col++) // for each attributes
	{
		pq->push_back(priority_queue<pds, vector<pds>, Custom_Comparator>()); // push_back a new priority queue for attribute-x
		
		if (DEBUG) cout << attributes->at(col).name << ":\n";

		Attribute* att = &(attributes->at(col));
		map<string, int>::iterator label_it = att->label.begin();
		double info_att = 0;
		while (label_it != att->label.end()) // for each attribute's label
		{
			double label_count = (double)label_it->second;
			double probability = label_count / total_data;
			map<pair<string, string>, int>::iterator class_tuple_it = att->class_tuple.begin();
			double info_label = 0;
			while (class_tuple_it != att->class_tuple.end()) // calculate Info(a,b,...) where "a,b,..." is according to the number of classifier label
			{
				if (class_tuple_it->first.first == label_it->first) // find the class_tuple with the same label
				{
					double n = class_tuple_it->second;
					info_label += -(n / label_count)*log2(n / label_count);
				}
				class_tuple_it++;
			}
			if(DEBUG) printf("%s:\t%F*%F+\n", label_it->first.c_str(), probability, info_label); // DEBUG
			info_att += probability * info_label;
			pq->at(col).push(make_pair(probability* info_label, label_it->first)); // pair<purity,label name>

			label_it++;
		}

		if(DEBUG) cout << ">>"<< info_parent << " - (" << info_att << ") = " << info_parent - info_att << endl; // DEBUG
		if(DEBUG) printf("\n\n\n"); // DEBUG;

		
		if ((info_parent - info_att) > maximum)
		{
			selected_att_idx = col;
			maximum = (info_parent - info_att);
		}
	}
}

void BUILD_TREE(vector<vector<string> >* dataset, DT_node* curr)
{
	// 1. read the header
	int col_sz = dataset->at(0).size(); // number of attribute (including classifier)
	Classifier* classifier = new Classifier(dataset->at(0)[col_sz - 1]); // create classifier
	vector<Attribute>* attributes = new vector<Attribute>(); // attributes=predictor (everything besides the classifier/target)
	for (int i = 0; i < col_sz -1; i++)
	{
		attributes->push_back(Attribute(dataset->at(0)[i])); // create attribute with the header name
	}

	// 2. read the dataset
	int row_sz = dataset->size();
	int total_data = row_sz - 1; // not including header
	read_dataset(dataset, classifier, attributes, row_sz, col_sz);
	
	if(DEBUG) DEBUG_attribute_count(attributes); // DEBUG

	// 3. find Info.parent
	double info_parent = 0;
	get_info_parent(classifier, total_data, info_parent);

	// 4. find MAXIMUM gain (MINIMUM Info.x)
	double maximum = -1;
	int selected_att_idx = -1;
	// this vector of priority queue is to sort the label based on its effect to the information gain
	// (used to determine the order of selection on the next building tree recursion)
	vector<priority_queue<pds,vector<pds>, Custom_Comparator > >* pq = // pair<double,string> = pair<information gain, label name>
		new vector<priority_queue<pds, vector<pds>, Custom_Comparator > >(); // pq[attribute-n] = priority queue for attribute-n
	pq->clear();
	get_maximum_gain(attributes, col_sz, total_data, info_parent, maximum, selected_att_idx, pq);
	curr->name = attributes->at(selected_att_idx).name;
	
	// set header to current header, and remove the column with the selected attribute's index
	vector<string> header = vector<string>(dataset->at(0));
	header.erase(header.begin() + selected_att_idx);

	// create dataset, DT_node child as much as the number of label for selected attribute
	Attribute* att = &(attributes->at(selected_att_idx));
	priority_queue<pds, vector<pds>, Custom_Comparator> attribute_pq = pq->at(selected_att_idx);
	while(!attribute_pq.empty())
	{
		string attribute_label = attribute_pq.top().second; // label name of attribute
		DT_node* child = new DT_node(); // create new child

		if (DEBUG) cout << "<<Created new child:" << attribute_label << ">>\n";

		curr->childs.push_back(make_pair(attribute_label, child)); // add <condition_name, next_child>
		if (
			(attribute_pq.top().first > 0) && (dataset->at(0).size()>1) && !(all_same_class(dataset))
			) // meaning that it contributes to the information gain AND there's still remaining attribute for further partitioning AND not all of the dataset is of the same class
		{
			// keep going/partitioning from the child
			vector<vector<string> >* next_dataset = new vector<vector<string> >(); // create new dataset for child
			next_dataset->clear();
			next_dataset->push_back(header);

			for (int i = 1; i < dataset->size(); i++)
			{
				// filter the dataset with attribute equals to the current label in the priority_queue
				if (dataset->at(i).at(selected_att_idx) == attribute_label)
				{
					//dataset->at(i).erase(dataset->at(i).begin() + selected_att_idx); // delete the column of the selected attribute's index
					next_dataset->push_back(dataset->at(i)); // add the line to the new dataset
					//dataset->erase(dataset->begin() + i); // remove the line from current dataset
					//i--; // fix the index after delete
				}
			}
			if (DEBUG) printf("<%s-> %s>\n\n", att->name.c_str(), attribute_label.c_str()); // DEBUG
			DT_build_queue.push(make_pair(next_dataset, child) ); // recursion to the newly-created child node (Breadth-First)
		}
		else
		{
			// find majority classifier label for the attribute label
			int col_sz = dataset->at(0).size();
			int row_sz = dataset->size();
			map<string, int>* count = new map<string, int>();
			count->clear();
			for (int i = 1; i < row_sz; i++)
			{
				if (dataset->at(i).at(selected_att_idx) == attribute_label)
				{
					string class_label = dataset->at(i).at(col_sz - 1);
					(*count)[class_label]++;
				}
			}
			
			map<string, int>::iterator it = count->begin();
			string class_label = "null";
			int maximum = -1;
			while (it != count->end()) // get the maximum/majority class label
			{
				if (it->second > maximum)
				{
					class_label = it->first;
					maximum = it->second;
				}
				it++;
			}
			child->class_label = class_label;
		}
		
		attribute_pq.pop();
	}

	delete classifier;
	delete attributes;
	delete pq;
}

string CHECK_DATA(int idx, map<string,int>* attribute_index, DT_node* curr)
{
	while (curr->childs.size() > 0) // as the last child will have child size of 0
	{
		string attribute_label = test_dataset[idx].at((*attribute_index)[curr->name]);
		bool found = false;
		for (int i = 0; i < curr->childs.size(); i++)
		{
			if (curr->childs.at(i).first == attribute_label)
			{
				found - true;
				curr = curr->childs.at(i).second;
				break;
			}
		}
		if (!found) break;
	}
	return curr->class_label;
}

void TEST_DATASET(DT_node* root)
{
	ifstream inputFile(test_name);
	string line;
	while (getline(inputFile, line))
	{
		vector<string> row; row.clear();
		string item; item.clear();
		stringstream ss = stringstream(line);
		while (ss >> item)
		{
			row.push_back(item);
		}
		test_dataset.push_back(row);
	}

	map<string, int> attribute_index; // to make searching attribute easier
	int col_sz = test_dataset[0].size();
	for (int i = 0; i < col_sz; i++)
	{
		attribute_index[test_dataset[0].at(i)] = i;
		out << test_dataset[0].at(i) << '\t'; // print header (attribute)
	}
	attribute_index[classifier_name]++;
	out << classifier_name << '\n'; // print header (classifier)

	int row_sz = test_dataset.size();
	for (int i = 1; i < row_sz; i++)
	{
		for (int j = 0; j < col_sz; j++)
		{
			out << test_dataset[i][j] << '\t'; // print attribute label
		}

		out << CHECK_DATA(i,&attribute_index,root) << '\n'; // print class label
	}
}

int main(int argc, char* argv[])
{
	training_name = string(argv[1]);
	test_name = string(argv[2]);
	output_name = string(argv[3]);

	scanFile();

	DT_node* root = new DT_node();

	DT_build_queue.push(make_pair(&database, root));
	while (!DT_build_queue.empty())
	{
		if (DEBUG) cout << "////////// BUILDING TREE //////////\n";
		BUILD_TREE(DT_build_queue.front().first, DT_build_queue.front().second);
		DT_build_queue.pop();
	}

	out.open(output_name);
	TEST_DATASET(root);
	out.close();

	// recursive delete root here (TODO)
	delete root;
	if(DEBUG) cout << "\n\n<<Press enter to exit>>";
	if(DEBUG) cin.get();
	return 0;
}