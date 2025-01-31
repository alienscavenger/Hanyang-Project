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
#include <stack>
#include <iterator>

using namespace std;
typedef pair<double, string> pds;

string training_name;
string test_name;
string output_name;
ofstream out;
string classifier_name; // to make finding the final label easier
bool DEBUG = false;
bool GINI = false;

class Classifier
{
public:
	string name; // attribute name
	map<string, int> label; // map<label name, count>
	Classifier(string name) : name(name) {}
};

class Attribute
{
public:
	string name; // attribute name
	map<string, int> label; // map<label name, count>
	map<pair<string, string>, int> class_tuple; // < <label, class_label>, occurence>
	Attribute(string name) : name(name) {};
};

class DT_node
{
public:
	string name; // attribute name (partition)
	string class_label; // class prediction
	vector<pair<string, DT_node*> > childs; // pair<condition, next_node>
	DT_node* parent;
	bool is_leaf; // is the last node (leaf)
	DT_node()
	{
		is_leaf = true;
		class_label = "<unknown>";
		parent = NULL;
	}
};

vector<vector<string> > database; 
vector<vector<string> > test_dataset;
queue < pair< vector<vector<string> >*, pair<DT_node*,string> > > DT_build_queue; // doesn't matter whether you use stack or queue, since each branch is INDEPENDENT

void scanFile()
{
	// scan the text file, and save it to a vector
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
			attributes->at(col).label[attribute_label]++; // increment the counter for the attribute's label
			attributes->at(col).class_tuple[make_pair(attribute_label, classifier_label)]++; // increment the label_class tuple counter
		}
	}
}

void DEBUG_attribute_count(vector<Attribute>* attributes, Classifier* classifier)
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

	map<string, int>::iterator  cit = classifier->label.begin();
	cout << classifier->name << ":\n";
	while (cit != classifier->label.end())
	{
		cout << "(" + cit->first + "," + to_string(cit->second) + ")" << '\t';
		cit++;
	}
	cout << "\n";

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
		info_parent += -(count / total_data)*log2((count / total_data)); // -Sigma[Pr*log2(Pr)]
		classifier_label_iter++;
	}
}

class Custom_Comparator // doesn't really matter, but makes debugging easier
{
public:
	bool operator()(pds a, pds b)
	{
		return a.first > b.first; //ascending (the lower the Info.x, the higher the Information Gain)
	}
};

void get_maximum_gain(vector<Attribute>* attributes, int col_sz, int total_data, double info_parent,
	int& selected_att_idx, vector < priority_queue<pds, vector<pds>, Custom_Comparator> >* pq)
{
	if(GINI) // If using Gini index instead of information gain (entropy)
	{
		double minimum = 99999;
		for (int col = 0; col < col_sz - 1; col++) // for each attributes
		{
			pq->push_back(priority_queue<pds, vector<pds>, Custom_Comparator>()); // push_back a new priority queue for attribute-x

			Attribute* att = &(attributes->at(col));
			map<string, int>::iterator label_it = att->label.begin();
			double gini_att = 0;
			while (label_it != att->label.end()) // for each attribute's label
			{
				double label_count = (double)label_it->second;
				double probability = label_count / total_data;
				map<pair<string, string>, int>::iterator class_tuple_it = att->class_tuple.begin();
				double gini_label = 1;
				while (class_tuple_it != att->class_tuple.end()) // calculate Info(a,b,...) where "a,b,..." is according to the number of classifier label
				{
					if (class_tuple_it->first.first == label_it->first) // find the class_tuple with the same label
					{
						double n = class_tuple_it->second;
						gini_label -= (n/label_count)*(n/label_count);
					}
					class_tuple_it++;
				}
				gini_att += probability * gini_label;

				pq->at(col).push(make_pair(probability * gini_label, label_it->first)); // pair<purity,label name>

				label_it++;
			}

			if (gini_att< minimum)
			{
				selected_att_idx = col;
				minimum = gini_att;
			}
		}
		return;
	}
	else
	{
		double maximum = -1;
		for (int col = 0; col < col_sz - 1; col++) // for each attributes
		{
			pq->push_back(priority_queue<pds, vector<pds>, Custom_Comparator>()); // push_back a new priority queue for attribute-x

			if (DEBUG) cout << attributes->at(col).name << ":\n";

			Attribute* att = &(attributes->at(col));
			map<string, int>::iterator label_it = att->label.begin();
			double info_att = 0;
			double split_info = 0;
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
						info_label += -(n / label_count) * log2(n / label_count); // -pi*log2(pi)
					}
					class_tuple_it++;
				}
				if (DEBUG) printf("%s:\t%F*%F+\n", label_it->first.c_str(), probability, info_label); // DEBUG
				info_att += probability * info_label; // Sigma[probability*Info(Dj)]
				split_info += -probability * log2(probability); //	-Sigma[Prob(D)*log2(Prob(D)]

				pq->at(col).push(make_pair(probability * info_label, label_it->first)); // pair<purity,label name>

				label_it++;
			}
			if (split_info < 1e-11) split_info = 1.0; // if split_info == 0, set it to 1

			if (DEBUG) cout << ">>" << info_parent << " - (" << info_att << ") = " << info_parent - info_att << " (split_info=" + to_string(split_info) + ")" << endl; // DEBUG
			if (DEBUG) printf("\n\n"); // DEBUG;

			if ((info_parent - info_att) / split_info > maximum) // find the maximum gain
			{
				selected_att_idx = col;
				maximum = (info_parent - info_att) / split_info;
			}
		}
		return;
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
	read_dataset(dataset, classifier, attributes, row_sz, col_sz); // initialize attributes' labels count
	
	if(DEBUG) DEBUG_attribute_count(attributes, classifier); // DEBUG

	// 3. find Info.parent
	double info_parent = 0;
	get_info_parent(classifier, total_data, info_parent);

	// 4. find MAXIMUM gain (MINIMUM Info.x)
	// this vector of priority queue is to sort the label based on its effect to the information gain (in the end, it's doesn't really matter)
	vector<priority_queue<pds,vector<pds>, Custom_Comparator > >* pq = // pair<double,string> = pair<information gain, label name>
		new vector<priority_queue<pds, vector<pds>, Custom_Comparator > >(); // pq[attribute-n] = priority queue for attribute-n
	pq->clear();
	int selected_att_idx = -1;
	get_maximum_gain(attributes, col_sz, total_data, info_parent, selected_att_idx, pq);
	curr->name = attributes->at(selected_att_idx).name; // current node name = selected attributes
	if (DEBUG) cout << "Selected: " << curr->name << "\n\n";

	// set header to current header, and remove the column with the selected attribute's index
	vector<string> header = vector<string>(dataset->at(0));
	header.erase(header.begin() + selected_att_idx);

	// 5. create dataset, DT_node child as much as the number of label for selected attribute
	Attribute* att = &(attributes->at(selected_att_idx));
	priority_queue<pds, vector<pds>, Custom_Comparator> attribute_pq = pq->at(selected_att_idx);
	while(!attribute_pq.empty()) // this contains all 'available' label (not necessarily everything, depending on current location on the branch node) of the selected attribute
	{
		string attribute_label = attribute_pq.top().second; // label name of attribute
		DT_node* child = new DT_node(); // create new child
		child->parent = curr; // set child's parent to current node

		if (DEBUG) cout << "<<Created new child:" << attribute_label << ">>\n";

		double contribution = attribute_pq.top().first; // attribute's label
		curr->childs.push_back(make_pair(attribute_label, child)); // add <condition_name, next_child> (add the new child to current node's child list)
		if (contribution>0 && (header.size()>1)) // meaning that it contributes to the information gain AND there's still remaining attribute for further partitioning AND not all of the dataset is of the same class
		{
			// 5.a keep going from the CHILD (PARTITION)
			child->is_leaf = false; // if it partition, that means that it is not a leaf
			vector<vector<string> >* next_dataset = new vector<vector<string> >(); // create new dataset for child
			next_dataset->clear();
			next_dataset->push_back(header);

			int row_sz = dataset->size();
			for (int i = 1; i < row_sz; i++)
			{
				// filter the dataset with attribute equals to the current label in the priority_queue
				if (dataset->at(i).at(selected_att_idx) == attribute_label)
				{
					next_dataset->push_back(dataset->at(i)); // add the line to the new dataset
					next_dataset->back().erase(next_dataset->back().begin() + selected_att_idx); // delete the column of the selected attribute's index so that it won't be picked next time
				}
			}
			if (DEBUG) printf("PUSH <%s-> %s>\n\n", att->name.c_str(), attribute_label.c_str()); // DEBUG
			DT_build_queue.push(make_pair(next_dataset, make_pair(child,attribute_label)) ); // recursion to the newly-created child node (Breadth-First)
		}

		// 6 find majority classifier label for the attribute label
		int col_sz = dataset->at(0).size();
		int row_sz = dataset->size();
		map<string, int>* count = new map<string, int>(); // count of class label
		count->clear();
		for (int i = 1; i < row_sz; i++)
		{
			if (DEBUG) cout << dataset->at(i).at(selected_att_idx) << ' ';
			if (dataset->at(i).at(selected_att_idx) == attribute_label) // only take the attribute label with the wanted one
			{
				string class_label = dataset->at(i).at(col_sz - 1);
				(*count)[class_label]++; // increse the class label count
			}
		}
		if (DEBUG) cout << "\n";

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
		if (DEBUG) cout << "<" + class_label + ">\n" << endl;
		child->class_label = class_label; // set CHILD's class label as the majority
		
		attribute_pq.pop();
		delete count;
	}

	delete classifier;
	delete attributes;
	delete pq;
}

string CHECK_DATA(int idx, map<string,int>* attribute_index, DT_node* curr) // classify the label
{
	while (true) // keep looping
	{
		string attribute_label = test_dataset[idx].at((*attribute_index)[curr->name]);
		if (DEBUG) cout << "test_dataset[" + to_string(idx) + "].at(" + "(*attribute_index)" + "[" + curr->name + "]) = " + attribute_label << endl;
		bool found = false;
		for (int i = 0; i < curr->childs.size(); i++) // loop through each node's child (and its condition)
		{
			if (DEBUG) cout << "\tcurr->chillds.at(" << i << ").first = " << curr->childs.at(i).first << endl;
			if (curr->childs.at(i).first == attribute_label) // find the appropriate condition
			{
				if (DEBUG) cout << "FOUND\n";
				found = true;
				curr = curr->childs.at(i).second; // go to next child
				break;
			}
		}
		if (!found) break; // if condition not found on any label, then use the majority class label of current attribute label
	}
	if (DEBUG) cout << "RETURN: " << curr->class_label << endl;
	return curr->class_label; // return current node class label
}

void TEST_DATASET(DT_node* root)
{
	ifstream inputFile(test_name);
	string line;
	while (getline(inputFile, line)) // get all the test dataset, and copy it to a local variable
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
	attribute_index[classifier_name] = col_sz-1;
	out << classifier_name << '\n'; // print header (classifier)

	int row_sz = test_dataset.size();
	for (int i = 1; i < row_sz; i++)
	{
		for (int j = 0; j < col_sz; j++)
		{
			out << test_dataset[i][j] << '\t'; // print attribute label
		}

		string res = CHECK_DATA(i, &attribute_index, root); // check the classified label
		out << res << '\n'; // print class label
	}
}

void DEBUG_build_tree(DT_node* root)
{
	printf("\n\n\nTREE\n");

	stack<pair< pair<DT_node*,int>,string> > st;
	st.push(make_pair(make_pair(root,0),"root"));
	while (!st.empty())
	{
		DT_node* curr = st.top().first.first;
		string cond = st.top().second;
		int tab = st.top().first.second;
		st.pop();

		for (int i = 0; i < tab; i++) cout << "|\t";
		cout << "(" + cond + ") ";
		if (curr->childs.size() == 0) cout << " -> " + curr->class_label;
		cout << endl;
		for (int i = 0; i < tab+1; i++) cout << "|\t";
		cout << curr->name << endl;

		for (int i = 0; i < curr->childs.size(); i++)
		{
			st.push(make_pair(make_pair(curr->childs[i].second, tab + 1), curr->childs[i].first));
		}

	}

	printf("\n\n\n");
}

int main(int argc, char* argv[])
{
	training_name = string(argv[1]);
	test_name = string(argv[2]);
	output_name = string(argv[3]);

	scanFile();

	DT_node* root = new DT_node(); // create the root
	root->is_leaf = false;

	DT_build_queue.push(make_pair(&database, make_pair(root,"root"))); // add the root
	while (!DT_build_queue.empty()) // breadth-first building decision tree
	{
		vector<vector<string> >* qdataset = DT_build_queue.front().first;
		DT_node* qnode = DT_build_queue.front().second.first;
		string qcondition = DT_build_queue.front().second.second; // used for debugging
		DT_build_queue.pop();

		if (DEBUG) cout << "////////// BUILDING TREE ////////// from:";
		if (DEBUG) cout << ((qnode->parent == NULL) ? "root" : qnode->parent->name) <<
			" " << qcondition << endl;
		BUILD_TREE(qdataset, qnode); // build the tree
	}

	if(DEBUG) DEBUG_build_tree(root); // DEBUG

	out.open(output_name);
	TEST_DATASET(root);
	out.close();

	// (TODO) recursive delete root here
	delete root;

	if(DEBUG) cout << "\n\n<<Press enter to exit>>";
	if(DEBUG) cin.get();
	return 0;
}