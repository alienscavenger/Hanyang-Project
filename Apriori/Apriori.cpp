// Apriori.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>

using namespace std;

vector<map<int, int> > database;
int databaseSize = -1;
int maxItemSize = -1; // max length/number of item for each transaction/line
int minSupport;
string inputName;
string outputName;
ofstream out;
bool printChar = false; //DEBUG
bool debugging = false; //DEBUG

class ItemSet
{
public:
	vector<int> items;
	int size;
	ItemSet(vector<int>items, int size) : items(items), size(size) {}
};
vector<vector<ItemSet> > frequent; // frequent[1] = frequent pattern of length-1
vector<ItemSet> candidate;

// UNUSED
//struct ItemSetCompare // custom comparator used for map data structure
//{
//	bool operator() (const ItemSet& lhs, const ItemSet& rhs) const
//	{
//		vector<int> lhs_items = lhs.items;
//		vector<int> rhs_items = rhs.items;
//		sort(lhs_items.begin(), lhs_items.end());
//		sort(rhs_items.begin(), rhs_items.end());
//		int sz = lhs_items.size();
//		for (int i = 0; i < sz; i++)
//		{
//			if (lhs_items[i] != rhs_items[i])
//			{
//				return lhs_items[i] < rhs_items[i];
//			}
//		}
//		return false;
//	}
//};
//map<ItemSet, int, ItemSetCompare> counter;
//void test_map_comparator() //DEBUG
//{
//	ItemSet one = ItemSet(vector<int>{1, 2}, 2);
//	ItemSet two = ItemSet(vector<int>{2, 3}, 2);
//	counter[one]++;
//	counter[two]++;
//
//	map<ItemSet, int>::iterator it = counter.begin();
//	while (it != counter.end())
//	{
//		for (unsigned i = 0; i < it->first.items.size(); i++)
//		{
//			cout << it->first.items[i] << ' ';
//		}
//		cout << '\n';
//		it++;
//	}
//}

void scanFile()
{
	ifstream inputFile(inputName);
	map<int, int> counter; // to count the support for length-1 (k=1) frequent item on ALL lines/whole database
	int itemCount = 0; // number of item in the database
	string line;

	while (getline(inputFile, line)) // get each line of the text input file
	{
		map<int, int> dataCounter; // used to save the counter of different items
		dataCounter.clear(); // clear for every iteration (when going to next line)

		stringstream ss(line);
		int item;
		int length = 0;
		while (ss >> item) // get each item on each line
		{
			length++;
			dataCounter[item]++;
			counter[item]++;
			itemCount++;
		}
		maxItemSize = max(length, maxItemSize);
		database.push_back(dataCounter); // insert the counter
	}
	minSupport = (minSupport*itemCount) / 100; // (minSupport/100) * itemCount
	if(debugging) minSupport = 2; //DEBUG
	databaseSize = database.size();

	//cout << "itemCount : " << itemCount << endl;
	//cout << "minSupport: " << minSupport << endl;

	map<int, int>::iterator it = counter.begin();

	frequent.push_back(vector<ItemSet>()); // for frequent[0] << this will be empty
	frequent.push_back(vector<ItemSet>()); // for frequent[1] << this will be for k=1

	while (it != counter.end()) //add k=1 item to the frequent pattern if it has sufficient support
	{
		if (it->second >= minSupport)
		{
			ItemSet iset = ItemSet(vector<int>{it->first}, 1);
			frequent[1].push_back(iset);
		}
		it++;
	}
}

void checkSuperset(int k)
{
	// check if we can find ALL subset for the candidate pattern on the frequent pattern list of length k-1
	vector<int> counter(candidate.size() + 1, 0);

	vector<ItemSet>::iterator it = frequent[k - 1].begin();
	while (it != frequent[k - 1].end()) // iterate through all frequent pattern of length k-1
	{
		//check each candidate,
		for (unsigned i = 0; i < candidate.size(); i++)
		{
			bool foundAll = true;
			//check if all item in the ITEMSET can be found in the CANDIDATE
			for (unsigned j = 0; j < it->items.size(); j++)
			{
				if (find(candidate[i].items.begin(), candidate[i].items.end(), it->items[j]) == candidate[i].items.end())
				{
					// one of the item in previous frequent pattern is not found
					foundAll = false;
					break;
				}
			}
			//if every item is found, then add the counter for the candidate index;
			if (foundAll)counter[i]++;
		}
		it++;
	}

	// number of ALL subset must be exactly (2^n)-2 (where n is the length of the itemset)
	// (minus 2 because we don't want empty subset and we don't want the whole itemset (n-Choose-0 and n-Choose-n)

	// number of subset of length k-1 must be equal to k (since nChoose(n-1) equal to n)
	for (unsigned i = 0; i < candidate.size(); i++)
	{
		if (counter[i] != k)
		{
			candidate.erase(candidate.begin() + i); // erase candidate if it doesn't found all k-1 subset
			counter.erase(counter.begin() + i); // erase the counter too
			i--; // decrease i to fix the looping index
		}
	}
}
void generateSuperset(map<int, int>* hashTable, vector<ItemSet>* candidate, vector<int>* lis1, vector<int>* lis2, int k)
{
	map<int, int> newList; // new candidate of size k
	vector<int>::iterator it;

	// add all the items from both item set (lis1 and lis2), and add the counter to the newList (ex: {1,3,5} + {5,3,4} = {1(x1), 3(x2), 4(x1), 5(x2)} )
	for (it = lis1->begin(); it != lis1->end(); it++) newList[*it]++;
	for (it = lis2->begin(); it != lis2->end(); it++) newList[*it]++;


	if (newList.size() == k) // only take it if the size the joining process is EXACTLY k (or another way to see it, lis1 and lis2 should have k-2 item in common)
	{
		/* //DEBUG
		map<int, int>::iterator itt = newList.begin();
		while (itt != newList.end())
		{
			printf("<%d>,", itt->first);
			itt++;
		}
		cout << endl;
		*/

		double hash = 1.0;
		double pi = atan(1) * 4; // just some constant to be used for the hash function
		map<int, int>::iterator mit = newList.begin();
		while (mit != newList.end()) // calculate hash number
		{
			hash += (pi * hash * mit->first * mit->first); // newList map is always sorted, so it doesn't matter what's the order of the item of the same set is inserted to the map
			hash = fmod(hash, 0x3f3f3f3f); // modulo so that it doesn't overflow
										   // this is probably not guaranteed to be unique for every combination of frequent pattern item,
										   // but it is good enough for current dataset
			mit++;
		}

		//printf("<%d>\n", (int)hash); //DEBUG
		hash = round(hash); // round it to int

		// check hash table to find duplicate item (if the same superset has been generated previously)
		if (hashTable->find((int)hash) != hashTable->end()) // if found duplicate
		{
			return;
		}
		(*hashTable)[(int)hash]++; // add to hash table if no same item set found

		//convert the map into a vector
		vector<int> itemLis;
		mit = newList.begin();
		while (mit != newList.end())
		{
			itemLis.push_back(mit->first);
			mit++;
		}
		candidate->push_back(ItemSet(itemLis, k)); // support is not yet known, so leave it at default
	}
}

void generateCandidate(int k)
{
	candidate.clear();
	int sz = frequent[k - 1].size(); // number of length k-1 pattern
	map<int, int>* hashTable = new map<int, int>(); // hash table so that we won't generate duplicate supersets
	hashTable->clear();

	// generate superset by self-joining two frequent pattern of length k-1
	for (int i = 0; i < sz - 1; i++)
	{
		for (int j = i + 1; j < sz; j++)
		{
			generateSuperset(hashTable, &candidate, &frequent[k - 1][i].items, &frequent[k - 1][j].items, k);
		}
	}
	checkSuperset(k);

	//printf("k=%d\ncandidateSize:%d\n", k, candidate.size());//DEBUG

	//for (unsigned i = 0; i < candidate.size(); i++) //DEBUG (PRINT CANDIDATE FREQUENT PATTERN)
	//{
	//	int sz = candidate[i].size;
	//	for (int j = 0; j < sz; j++)
	//	{
	//		cout << candidate[i].items[j] << ' ';
	//	}
	//	cout << endl;
	//}
}

void checkCandidate(int k) //count support for candidate, then add to frequent pattern vector if >= minSupport
{
	int sz = database.size();
	vector<int>counter(candidate.size(), 0); //initialize counter of vector the size of candidate with zeros

											 // INCREASE COUNTER FOR EACH CORRECT PATTERN
	for (int i = 0; i < sz; i++)// check each line of the database
	{
		for (unsigned j = 0; j < candidate.size(); j++)
		{
			bool add = true;
			for (int l = 0; l < k; l++)//check if all item in the frequent pattern candidate can be found on the item list
			{
				if (database[i].find(candidate[j].items[l]) == database[i].end()) //at least 1 item is not found on the transaction's item list
				{
					add = false;
					break;
				}
			}
			if (add) counter[j]++; //if everything is found, add to the counter (support)
		}
	}

	bool create = true;
	for (unsigned i = 0; i < candidate.size(); i++)
	{
		//printf("counter[%d]:%d\n", i, counter[i]); //DEBUG (PRINT SUPPORT FOR A CANDIDATE)
		
		if (counter[i] >= minSupport)
		{
			if (create) //create the frequent[k] vector, but only one time, and only once at least one frequent pattern is found
			{
				frequent.push_back(vector<ItemSet>());
				create = false;
			}
			frequent[k].push_back(candidate[i]); // add candidate pattern-i to the list of frequent pattern length-k
		}
	}

	//for (int i = 0; i < frequent[k].size(); i++) //DEBUG (PRINT FREQUENT PATTERN OF SIZE-K)
	//{
	//	for (int j = 0; j < k; j++)
	//	{
	//		printf("%d, ", frequent[k][i].items[j]);
	//	}
	//	cout << endl;
	//}
}

void argumentInitialize(int argc, char* argv[])
{
	minSupport = atoi(argv[1]);
	inputName = string(argv[2]);
	outputName = string(argv[3]);
}

void printFP(int k) //DEBUG
{
	if (frequent.size() <= k) return; // ex: if frequent size is 4 (index only reach 3), then k should not be 4 or higher

	cout << "\nPRINTING k-" << k << " FREQUENT PATTERN" << endl;
	for (unsigned j = 0; j < frequent[k].size(); j++)
	{
		for (int l = 0; l < frequent[k][j].size; l++)
		{
			cout << (l > 0 ? " " : "");
			cout << frequent[k][j].items[l];
		}
		cout << endl;
	}
}

float getSupport(vector<int>* items, int k)
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
	return (float)(counter*100.0f) / databaseSize;
}

float getConfidence(vector<int>* left, vector<int>* right)
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
	return (containsBoth*100.0f) / containsLeft;
}

void printSet(vector<int>* set)
{
	out << '{';
	int sz = set->size();
	for (int i = 0; i < sz - 1; i++)
	{
		if (printChar)
			out << 'A' + set->at(i) - 1;
		else
			out << set->at(i);
	}
	if (printChar)
		out << 'A' + set->at(sz - 1) - 1;
	else
		out << set->at(sz - 1);
	out << '}';
}

void getAssociative(int k) // k starts at 2
{
	if (frequent.size() <= k) return;
	int sz = frequent[k].size();
	for (int i = 0; i < sz; i++) // loop for every frequent pattern of size k
	{
		vector<int>* items = &frequent[k][i].items; // for easier typing
		float support = getSupport(items, k);
		sort(items->begin(), items->end()); // it should be sorted in the first place, but just to be safe (for doing permutation)
		do
		{
			for (int separator = 0; separator < (k - 1); separator++) // separator is the separator between the 'item' and 'associative item'
			{
				// [index 0,separator], (separator,index k-1]
				// {left set} (separator) {right set}
				// {item_set} (separator) {associative_item_set}

				vector<int> item_set = vector<int>(); item_set.clear();
				vector<int> associative_item_set = vector<int>(); associative_item_set.clear();
				for (int i = 0; i <= separator; i++)
				{
					item_set.push_back((*items)[i]);
				}
				for (int i = separator + 1; i < k; i++)
				{
					associative_item_set.push_back((*items)[i]);
				}
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
			}
		} while (next_permutation(items->begin(), items->end()));
	}
}

int main(int argc, char* argv[])
{
	argumentInitialize(argc, argv);
	scanFile();
	//printf("minimum support: %d\n", minSupport); //DEBUG

	int k = 1;
	//printFP(k);//DEBUG
	//printf("k=%d\ncandidateSize:%d\n", k, frequent[1].size());//DEBUG
	out.open(outputName);
	do
	{
		k++; // starts on k=2
		generateCandidate(k);
		checkCandidate(k);
		//printFP(k);//DEBUG
		getAssociative(k);
	} while (k<maxItemSize && frequent.size() > k &&frequent[k].size() > 1);
			// ex: when k is 4, size of frequent must be 5 (meaning that you can access frequent[4],
			// which also means that at least one frequent pattern has the length of 4, since
			// frequent[4] can only exist of at least one fp is found. (see: function checkCandidate() )
			// The size of frequent[k] must also be >1, so that at least
			// there is a possibility to create (at least) one superset from it.
			// k<maxItemSize, and not <=, because the logic check is before k is incremented (not after)

	//cout << "\nPRESS ENTER TO QUIT";
	//cin.get();
	out.close();
	return 0;
}