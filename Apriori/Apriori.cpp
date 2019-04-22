#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

using namespace std;

vector<map<int, int> > database; // keep the counter (and occurence) of items on each line (transaction) of the input file
int databaseSize = -1;
float minSupport;
string inputName;
string outputName;
ofstream out;

bool printChar = false; //DEBUG
bool debugging = false; //DEBUG

class ItemSet
{
public:
	vector<int> items;
	int size; // length of k
	ItemSet(vector<int>items, int size) : items(items), size(size) {}
};
vector<vector<ItemSet> > frequent; // frequent[x] = frequent pattern of length x
								   // example: when k is 4, size of frequent must be 5
vector<ItemSet> candidate; // temporary vector to save the candidate frequent pattern

void scanFile()
{
	ifstream inputFile(inputName); // uses C++ style filestream
	map<int, int> counter;	// used to count the support for length 1 (k=1) frequent item on ALL lines/whole database
							// map<item number, counter of the item>

	int itemCount = 0; // number of item in the database (of ALL transaction)
	string line;
	while (getline(inputFile, line)) // get each line of the text input file (into the 'line' variable)
	{
		map<int, int> dataCounter;	// used to save the counter of different items for each line/transaction
									// map<item number, counter of the item>
		dataCounter.clear(); // clear for every iteration (when going to next line)

		stringstream ss(line);
		int item;
		int length = 0; // number of item for each line
		while (ss >> item) // read each item on each line, and increment its counter
		{
			length++;
			dataCounter[item]++;
			counter[item]++;
			itemCount++;
		}
		database.push_back(dataCounter); // insert the counter
	}
	if (debugging) minSupport = 16; //DEBUG
	databaseSize = database.size();

	map<int, int>::iterator it = counter.begin();

	frequent.push_back(vector<ItemSet>());	// for frequent[0] (this will be empty)
	frequent.push_back(vector<ItemSet>());	// for frequent[1] (this will be for k=1)
											// (these two lines are needed, so that I can access frequent[1], since I wanted to start the index on '1')

	//now, add k=1 item to the frequent pattern if it has sufficient support
	while (it != counter.end())
	{
		float supp = ((it->second) * 100.0f) / databaseSize; // in percentage
		if (supp >= minSupport)
		{
			ItemSet iset = ItemSet(vector<int>{it->first}, 1); // create an object of ItemSet
			frequent[1].push_back(iset); // push the object to the frequent pattern (of length 1) list
		}
		it++;
	}
}

////////////////////////////////////////////////////////////////

void checkSuperset(int k)
{
	// check if we can find ALL k-1 SUBSET for the candidate pattern on the frequent pattern list (of length k-1)
	vector<int> counter(candidate.size() + 1, 0);

	vector<ItemSet>::iterator it = frequent[k - 1].begin();
	while (it != frequent[k - 1].end()) // iterate through all frequent pattern of length k-1
	{
		//for each candidate,
		for (unsigned i = 0; i < candidate.size(); i++)
		{
			//check if all item in the ITEMSET can be found in the CANDIDATE
			bool foundAll = true;
			for (unsigned j = 0; j < it->items.size(); j++)
			{
				// one of the item in previous frequent pattern is not found
				if (find(candidate[i].items.begin(), candidate[i].items.end(), it->items[j]) == candidate[i].items.end())
				{
					foundAll = false;
					break;
				}
			}
			// if every item is found, then add the counter for the candidate index. Otherwise, go to the next candidate
			if (foundAll)counter[i]++;
		}
		it++;
	}

	// number of subset of length k-1 must be equal to k (since n.Choose.(n-1) equal to n)
	for (unsigned i = 0; i < candidate.size(); i++)
	{
		if (counter[i] != k)
		{
			candidate.erase(candidate.begin() + i); // ERASE CANDIDATE if it doesn't found all its k-1 subset
			counter.erase(counter.begin() + i); // erase the counter too
			i--; // decrease i to fix the looping index
		}
	}
}

void generateSuperset(map<string, int> * hashTable, vector<ItemSet> * candidate, vector<int> * lis1, vector<int> * lis2, int k)
{
	// uses c++ STL instead of custom algorithm for the joining process
	sort(lis1->begin(), lis1->end()); // ascending order
	sort(lis2->begin(), lis2->end());
	vector<int> newList; // new candidate of size k
	merge(lis1->begin(), lis1->end(), lis2->begin(), lis2->end(), back_inserter(newList));
	sort(newList.begin(), newList.end()); // shouldn't make a difference
	newList.erase(unique(newList.begin(), newList.end()), newList.end()); // delete the duplicates elements of the newList

	if (newList.size() == k) // only take it if the size the joining process is EXACTLY k
							 // (or another way to see it, lis1 and lis2 should ONLY have k-2 item in common)
	{
		string hash("");
		for (int i = 0; i < k; i++)
		{
			hash += to_string(newList[i]); // create a hash key
			hash += "#";
		}
		if (hashTable->count(hash) > 0) return; // if hash already exist, return
		else hashTable->insert(make_pair(hash, 1)); // else, add to the hash table counter
		candidate->push_back(ItemSet(newList, k)); // push the joined set to the candidate vector
	}
}

void generateCandidate(int k)
{
	candidate.clear();
	int sz = frequent[k - 1].size(); // number of length k-1 pattern

	map<string, int>* hashTable = new map<string, int>(); // hash table so that we won't generate duplicate supersets
	hashTable->clear();

	// generate superset by self-joining two frequent pattern of length k-1
	for (int i = 0; i < sz - 1; i++)
	{
		for (int j = i + 1; j < sz; j++)
		{
			generateSuperset(hashTable, &candidate, &frequent[k - 1][i].items, &frequent[k - 1][j].items, k);
		}
	}
	free(hashTable);
	checkSuperset(k);
}

////////////////////////////////////////////////////////////////

void checkCandidate(int k) //count support for candidate, then add to frequent pattern vector if >= minSupport
{
	int sz = database.size();
	vector<int>counter(candidate.size(), 0); //initialize counter of vector the size of candidate with zeros

	// INCREASE COUNTER FOR EACH CORRECT PATTERN
	for (int line = 0; line < sz; line++)// check each line of the database
	{
		for (unsigned i = 0; i < candidate.size(); i++)
		{
			bool add = true;
			for (int item = 0; item < k; item++)//check if all item in the candidate's item list can be found on the item list
			{
				if (database[line].find(candidate[i].items[item]) == database[line].end()) //at least 1 item is not found on the transaction's item list
				{
					add = false;
					break; // stop checking
				}
			}
			if (add) counter[i]++; //if everything is found, add to the counter for the candidate frequent pattern (for support)
		}
	}

	frequent.push_back(vector<ItemSet>()); // push a vector to the frequent vector, so we can access frequent[k]
	for (unsigned i = 0; i < candidate.size(); i++)
	{
		float supp = (counter[i] * 100.0f) / databaseSize;
		if (supp >= minSupport)
		{
			frequent[k].push_back(candidate[i]); // add candidate pattern-i to the list of frequent pattern length-k
		}
	}
}

////////////////////////////////////////////////////////////////

float getSupport(vector<int> * items, int k)
{
	int counter = 0;
	for (int i = 0; i < databaseSize; i++)
	{
		bool foundAll = true;
		for (int j = 0; j < k; j++)
		{
			if (database[i].find((*items)[j]) == database[i].end())
			{
				foundAll = false;
				break;
			}
		}
		if (foundAll) counter++; // if every item in the items vector is found on the transaction, increment the counter
	}
	return (float)(counter * 100.0f) / databaseSize;
}

float getConfidence(vector<int> * left, vector<int> * right)
{
	int containsLeft = 0;
	int containsBoth = 0;

	for (int i = 0; i < databaseSize; i++)
	{
		bool foundAll = true;
		//check left data
		for (int j = 0; j < left->size(); j++)
		{
			if (database[i].find((*left)[j]) == database[i].end())
			{
				foundAll = false;
				break;
			}
		}
		if (!foundAll) continue; // if not all of it is found, continue to next transaction
		containsLeft++;

		//now check right data
		foundAll = true;
		for (int j = 0; j < right->size(); j++)
		{
			if (database[i].find((*right)[j]) == database[i].end())
			{
				foundAll = false;
				break;
			}
		}
		if (!foundAll) continue;
		containsBoth++;
	}
	if (containsLeft == 0) return -1.0f; //DEBUG
	return (containsBoth * 100.0f) / containsLeft;
}

void printSet(vector<int> * set)
{
	out << '{';
	int sz = set->size();
	for (int i = 0; i < sz - 1; i++) // with comma
	{
		if (printChar)
			out << (char)('A' + set->at(i) - 1);
		else
			out << set->at(i);
		out << ',';
	}
	if (printChar)
		out << (char)('A' + set->at(sz - 1) - 1);
	else
		out << set->at(sz - 1);
	out << '}';
}

vector<int> item_set;
vector<int> associative_item_set;

void getCombination(int count, int curr, int combiSize, int k, float support, vector<int> * items)
{
	if (count == combiSize) // if the combination size (of the left items) is equal to combiSize, then print the item set (left) and its association (right)
	{
		float confidence = getConfidence(&item_set, &associative_item_set);
		printSet(&item_set);
		out << '\t';
		printSet(&associative_item_set);
		out << '\t';

		out << fixed;
		out << setprecision(2) << support;
		out << '\t';
		out << setprecision(2) << confidence;
		out << '\n';
		return; // return, since we don't want to find any other combination of size > combiSize
	};

	for (int i = curr; i < k; i++) // items size will be always k
	{
		item_set.push_back(items->at(i));
		associative_item_set.erase(find(associative_item_set.begin(), associative_item_set.end(), items->at(i))); // works if there are no duplicate items
		// associative_item_set is only used for printing the association (the items which are NOT in the item_set)

		getCombination(count + 1, i + 1, combiSize, k, support, items); // nested loop recursion, with 'i' starting 1 value ahead of current one

		associative_item_set.push_back(items->at(i)); // push back the deleted item, it doesn't matter at what index, since during the erasing, it will uses 'find' anyway
		item_set.pop_back(); // the last inserted item will always be at the top/last index
	}
}

void getAssociative(int k) // k starts at 2
{
	if (frequent.size() <= k) return;
	int sz = frequent[k].size();
	for (int i = 0; i < sz; i++) // loop for every frequent pattern of size k
	{
		vector<int>* items = &frequent[k][i].items; // for easier typing
		float support = getSupport(items, k); // support is the same for every combination of association of the same item set

		item_set.clear();
		associative_item_set.clear();

		for (int i = 0; i < items->size(); i++) associative_item_set.push_back((*items)[i]); // put all item to the associative item set (right side)
		for (int j = 1; j < k; j++) // get combination of size 1 until (k-1) for the item set (left side)
		{
			getCombination(0, 0, j, k, support, items); // k Choose 1 until k Choose (k-1)
		}
	}
}

////////////////////////////////////////////////////////////////

void argumentInitialize(int argc, char* argv[])
{
	minSupport = atof(argv[1]);
	inputName = string(argv[2]);
	outputName = string(argv[3]);
}

int main(int argc, char* argv[])
{
	argumentInitialize(argc, argv);
	scanFile();
	//at this point, we should have the list of legth 1 frequent pattern
	int k = 1; // 'k' is the length of frequent pattern
	out.open(outputName);
	do
	{
		k++; // starts on k=2
		generateCandidate(k);
		checkCandidate(k);
		getAssociative(k);
	} while (frequent[k].size() > 1);
	// The size of frequent[k] must be >1, so that at least
	// there is a possibility to create (at least) one superset from it.
	out.close();
	return 0;
}
