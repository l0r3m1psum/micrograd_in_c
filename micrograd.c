// https://www.youtube.com/watch?v=VMj-3S1tku0
// https://github.com/karpathy/micrograd

#include <stddef.h>
#include <stdbool.h>
#include <assert.h>

#ifdef __APPLE__
#define BREAKPOINT __builtin_debugtrap()
#define NODISCARD __attribute__((warn_unused_result))
#else
#define BREAKPOINT
#define NODISCARD
#endif

#define ARRAY_LEN(a) (sizeof (a) / sizeof *(a))

#pragma mark Random Number Generator

#include <stdlib.h>

// TODO: use a better and faster generaror.
// https://stackoverflow.com/questions/62020542/generate-random-double-number-in-range-0-1-in-c
// https://stackoverflow.com/questions/35117014/generating-a-random-uniformly-distributed-real-number-in-c
// https://codereview.stackexchange.com/questions/159604/uniform-random-numbers-in-an-integer-interval-in-c
// https://www.cs.yale.edu/homes/aspnes/pinewiki/C(2f)Randomization.html
// https://people.sc.fsu.edu/~jburkardt/c_src/uniform/uniform.html

static void
RNG_set_seed(unsigned seed) {
	srand(seed);
}

// Generates a random number between [0,1).
static double
RNG_uniform01() {
	return (double)rand() / RAND_MAX;
}

// Generates a random number in (-1,1).
static double
RNG_uniform_m1p1() {
	return RNG_uniform01() * 2 - 1;
}

#pragma mark Memory Allocator

typedef struct Arena {
	void  *data;
	size_t size;
	size_t used;
} Arena;

#define Arena_FROM_ARRAY(a) \
	((Arena){.data = (a), .size = sizeof (a), .used = 0})

static bool
Arena_invariant(const Arena alloc[static 1]) {
	assert(alloc != NULL);
	if (alloc->data == NULL) {
		return (alloc->size == 0) & (alloc->used == 0);
	}
	return alloc->used <= alloc->size;
}

// NOTE: the operation like this is a bit unsafe in the sense that we could
// forget that we have created a sub allocator (with from_unsused) and keep
// using the original arena (which effectivelly should have all of it's memory
// used). This can create some subtle memory corruption bugs. A possible
// solution can be to set alloc->used = alloc->size when we create the sub
// allocator to "exahust" the memory in the original allocator and then use
// anoter method to give back said memeory (something like return_unsused). In
// this metod we would just set alloc->size -= unused->size and unused->size =
// unused->used = 0 (just to be sure). This is all great but if we start
// stacking sub allocators on top of sub allocatos keeping a record of the order
// in which operations have to be performed could be beneficial. Something like
// a static stack (of size N) that keeps track of all allocations contexts.
static Arena
Arena_from_unused(const Arena alloc[static 1]) {
	assert(Arena_invariant(alloc));

	Arena res = {
		.data = alloc->data + alloc->used,
		.size = alloc->size - alloc->used,
		.used = 0
	};

	assert(Arena_invariant(&res));
	return res;
}

static void *
Arena_alloc(Arena alloc[static 1], size_t req_size) {
	assert(Arena_invariant(alloc));
	if (alloc->size - alloc->used < req_size) {
		return NULL;
	}
	void *res = alloc->data + alloc->used;
	alloc->used += req_size;
	return res;
}

static void
Arena_reset(Arena alloc[static 1]) {
	assert(Arena_invariant(alloc));
	alloc->used = 0;
	assert(Arena_invariant(alloc));
}

#pragma mark Directed Acyclic Graph

// NOTE: Probably would be better to use an pool allocator since we only care
// about allocating the Value struct. This would also allow to use integer
// handles instead of pointer enabling us to use realloc to increase the memory
// usage. Also the allocator can perfectly be a global variable.

// TODO: avoid recursion in general.

// Primitive operations.
typedef enum Op {
	OP_NOP, // NOTE: another name could be OP_NUM or OP_PARAM.
	OP_ADD,
	OP_MUL,
	OP_TANH,
	OP_EXP,
	OP_POW,
} Op;

typedef struct Value {
	double data;
	double grad;
	struct Value *children0;
	// We determine which one is the active member with the op field.
	union {
		struct Value *children1;
		double const_arg;
	};
	Op op;
} Value;

// NOTE: I should be able to assert that we have a DAG by checking that the
// address of the childres should always be strictly less that the one of val
// (i.e. they have to be allocated before val). This also solves the problem of
// self loops. In the C standard comparing pointers is not always defined
// behavior but I think is always fine in modern machines, and in any case we
// use an arena allocator that implies a contigous block of memory.
static bool
Value_invariant(const Value *val) {
	if (val == NULL) {
		return true;
	}
	// If it is a POW operation it must have a child and it must not be itself.
	if (val->op == OP_POW) {
		return (val->children0 != NULL) & (val != val->children0);
	}
	// If only one of them is NULL it has to be the second one.
	if ((val->children0 == NULL) & (val->children1 != NULL)) {
		return false;
	}
	// If it is a no op it must be a source vertex.
	if ((val->op == OP_NOP) & ((val->children0 != NULL) | (val->children1 != NULL))) {
		return false;
	}
	// Self loops are no allowed in general.
	if ((val == val->children0) | (val == val->children1)) {
		return false;
	}
	return true;
}

static Value *
Value_new_internal(
		Arena alloc[static 1],
		double data,
		Value *children0,
		Value *children1,
		Op op
	) {
	assert(Arena_invariant(alloc));

	Value *res = Arena_alloc(alloc, sizeof *res);
	if (res == NULL) {
		return NULL;
	}

	res->data = data;
	res->grad = 0;
	res->children0 = children0;
	res->children1 = children1;
	res->op = op;

	assert(Value_invariant(res));
	return res;
}

static Value *
Value_new(Arena *alloc, double data) {
	return Value_new_internal(alloc, data, NULL, NULL, OP_NOP);
}

static Value *
Value_add(Arena alloc[static 1], Value *lhs, Value *rhs) {
	assert(Arena_invariant(alloc));
	assert(Value_invariant(lhs));
	assert(Value_invariant(rhs));

	if ((lhs == NULL) | (rhs == NULL)) {
		return NULL;
	}

	return Value_new_internal(alloc, lhs->data + rhs->data, lhs, rhs, OP_ADD);
}

static Value *
Value_mul(Arena alloc[static 1], Value *lhs, Value *rhs) {
	assert(Arena_invariant(alloc));
	assert(Value_invariant(lhs));
	assert(Value_invariant(rhs));

	if ((lhs == NULL) | (rhs == NULL)) {
		return NULL;
	}

	return Value_new_internal(alloc, lhs->data * rhs->data, lhs, rhs, OP_MUL);
}

#include <tgmath.h>

static Value *
Value_tanh(Arena alloc[static 1], Value *val) {
	assert(Arena_invariant(alloc));
	assert(Value_invariant(val));

	if (val == NULL) {
		return NULL;
	}

	return Value_new_internal(alloc, tanh(val->data), val, NULL, OP_TANH);
}

// NOTE: this can be implemented in terms this in terms of POW.
static Value *
Value_exp(Arena alloc[static 1], Value *val) {
	assert(Arena_invariant(alloc));
	assert(Value_invariant(val));

	if (val == NULL) {
		return NULL;
	}

	return Value_new_internal(alloc, exp(val->data), val, NULL, OP_EXP);
}

static Value *
Value_pow(Arena alloc[static 1], Value *val, double exponent) {
	assert(Arena_invariant(alloc));
	assert(Value_invariant(val));

	if (val == NULL) {
		return NULL;
	}

	Value *res = Arena_alloc(alloc, sizeof *res);
	if (res == NULL) {
		return NULL;
	}

	res->data = pow(val->data, exponent);
	res->grad = 0;
	res->children0 = val;
	res->const_arg = exponent;
	res->op = OP_POW;

	assert(Value_invariant(res));
	return res;
}

#pragma mark Non-Elementary Operations

static Value *
Value_neg(Arena alloc[static 1], Value *val) {
	// Karpathy does this by creating a new object. I this really necessary or
	// can I use a multiplication by constant operation?
	return Value_mul(alloc, val, Value_new(alloc, -1));
}

static Value *
Value_sub(Arena alloc[static 1], Value *lhs, Value *rhs) {
	return Value_add(alloc, lhs, Value_neg(alloc, rhs));
}

static Value *
Value_recip(Arena alloc[static 1], Value *val) {
	return Value_pow(alloc, val, -1);
}

static Value *
Value_div(Arena alloc[static 1], Value *lhs, Value *rhs) {
	return Value_mul(alloc, lhs, Value_recip(alloc, rhs));
}

#pragma mark Graph Visits

// TODO: I should be able to implement all this functions with a simple DFS that
// calls a callback + userdata.

// We do a naive DFS that considers a DAG as a compressed tree, therefore nodes
// can be visited/counted multiple times.
static void
Value_backprop_internal_count_max(Value *val, size_t count[static 1]) {
	assert(Value_invariant(val));
	assert(count != NULL);

	if (val == NULL) {
		return;
	}

	Value_backprop_internal_count_max(val->children0, count);
	if (val->op != OP_POW) {
		Value_backprop_internal_count_max(val->children1, count);
	}
	(*count)++;
}

// The topological sorting is performed using a reverse post-order DFS.
// https://algs4.cs.princeton.edu/42digraph/#:~:text=Remarkably%2C%20a%20reverse%20postorder%20in%20a%20DAG%20provides%20a%20topological%20order
// A reverse post-order DFS is not the seme as a pre-order DFS as Wikipedia
// explains well:
// https://en.wikipedia.org/wiki/Depth-first_search#Vertex_orderings#:~:text=Thus%20the%20possible%20preorderings%20are%20A%20B%20D%20C%20and%20A%20C%20D%20B%2C%20while%20the%20possible%20postorderings%20are%20D%20B%20C%20A%20and%20D%20C%20B%20A%2C%20and%20the%20possible%20reverse%20postorderings%20are%20A%20C%20B%20D%20and%20A%20B%20C%20D
static void
Value_backprop_internal_toposort(
		Value *val,
		size_t len[static 1],
		Value **vals,
		size_t visited_len[static 1],
		Value **visited
	) {
	assert(Value_invariant(val));

	if (val == NULL) {
		return;
	}

	// TODO: implement a proper hashset.
	for (size_t i = 0; i < *visited_len; i++) {
		if (val == visited[i]) {
			return;
		}
	}
	visited[(*visited_len)++] = val;

	Value_backprop_internal_toposort(val->children0, len, vals, visited_len, visited);
	if (val->op != OP_POW) {
		Value_backprop_internal_toposort(val->children1, len, vals, visited_len, visited);
	}
	vals[(*len)++] = val;
}

static bool
Value_is_toposorted_reversed(size_t len, Value *vals[static len]) {
	// We have to check that edge only go backward because we are testing for a
	// reversed topological order.
	for (size_t i = len; i --> 0;) {
		if (vals[i] == NULL) {
			continue;
		}
		if (vals[i]->children0 == NULL) {
			// This is a leaf node.
			continue;
		}
		for (size_t j = i+1; j < len; j++) {
			assert(vals + j >= vals + i);
			if (vals[j] == NULL) {
				continue;
			}
			if ((vals[i]->children0 == vals[j])
				| (vals[i]->op != OP_POW ? (vals[i]->children1 == vals[j]) : false)) {
				return false;
			}
		}
	}
	return true;
}

// NOTE: instead of doing a topological sort every time I could store the nodes
// in memory already sorted so that a simple for would allow me to calculate
// the gradient. This could be done by modifing the function that creates the
// node to insert the nodes in a sorted way.

// We do not need to do a general DFS for our topological sort (i.e. visiting
// even nodes that are not reachable from the starting one), since we only care
// about calculating the gradient of the nodes that we can reach from our
// source node (i.e. the loss) which ideally should be a sink. If you have more
// than one loss that you want to calculate (or in general back propagation
// that involve a common part of the graph), remember to zero the gradients
// between your back propagations.
static NODISCARD bool
Value_backprop(Arena alloc[static 1], Value *val) {
	assert(Arena_invariant(alloc));
	assert(Value_invariant(val));

	if (val == NULL) {
		return true;
	}

	size_t max_len = 0;
	Value_backprop_internal_count_max(val, &max_len);
	assert(max_len >= 1);
	Arena temp = Arena_from_unused(alloc);
	Value **vals = Arena_alloc(&temp, sizeof *vals * max_len);
	if (vals == NULL) {
		Arena_reset(&temp);
		return false;
	}
	Value **visited = Arena_alloc(&temp, sizeof *visited * max_len);
	if (visited == NULL) {
		Arena_reset(&temp);
		return false;
	}

	size_t vals_len = 0;
	size_t visited_len = 0;
	Value_backprop_internal_toposort(val, &vals_len, vals, &visited_len, visited);
	assert(Value_is_toposorted_reversed(vals_len, vals));
	assert(vals_len == visited_len);
	assert(vals[vals_len-1] == val);
	val->grad = 1;
	for (size_t i = vals_len; i --> 0;) {
		switch (vals[i]->op) {
		case OP_NOP: break;
		case OP_ADD:
			vals[i]->children0->grad += /* 1 * */ vals[i]->grad;
			vals[i]->children1->grad += /* 1 * */ vals[i]->grad;
			break;
		case OP_MUL:
			vals[i]->children0->grad += vals[i]->children1->data * vals[i]->grad;
			vals[i]->children1->grad += vals[i]->children0->data * vals[i]->grad;
			break;
		case OP_TANH:
			vals[i]->children0->grad += (1 - pow(vals[i]->data, 2)) * vals[i]->grad;
			break;
		case OP_EXP:
			vals[i]->children0->grad += vals[i]->data * vals[i]->grad;
			break;
		case OP_POW: {
			double exponent = vals[i]->const_arg;
			vals[i]->children0->grad += exponent
				* pow(vals[i]->children0->data, exponent - 1) * vals[i]->grad;
			break;
		}
		}
	}

	Arena_reset(&temp);
	return true;
}

#include <stdio.h>
#include <inttypes.h>

static bool
Value_print_dot_internal(const Value *val) {
	assert(Value_invariant(val));
	const char *node_stmt = "\t\"0x%" PRIXPTR "\" "
		"[label=\"%s{data: %lf | grad: %lf}\",tooltip=\"0x%" PRIXPTR "\"];\n";
	const char *const_node_stmt = "\t\"0x%" PRIXPTR "\" "
		"[label=\"%lf\",tooltip=\"0x%" PRIXPTR "\"];\n";
	const char *edge_stmt = "\t\"0x%" PRIXPTR "\" -> \"0x%" PRIXPTR "\";\n";

	if (val == NULL) {
		return true;
	}

	const char *op = NULL;
	switch (val->op) {
	case OP_NOP: op = ""; break;
	case OP_ADD: op = "+ | "; break;
	case OP_MUL: op = "* | "; break;
	case OP_TANH: op = "tanh | "; break;
	case OP_EXP: op = "exp | "; break;
	case OP_POW: op = "pow | "; break;
	}
	uintptr_t address = (uintptr_t)val;

	if (printf(node_stmt, address, op, val->data, val->grad, address) < 0) {
		return false;
	}
	if (val->op == OP_NOP) {
		return true;
	}

	if (printf(edge_stmt, address, (uintptr_t)val->children0) < 0) {
		return false;
	}
	if (!Value_print_dot_internal(val->children0)) {
		return false;
	}

	if (val->op == OP_POW) {
		uintptr_t id = (uintptr_t)&val->const_arg;
		if (printf(edge_stmt, address, id) < 0) {
			return false;
		}
		printf(const_node_stmt, id, val->const_arg, id);
	} else if (val->children1 != NULL) {
		if (printf(edge_stmt, address, (uintptr_t)val->children1) < 0) {
			return false;
		}
		if (!Value_print_dot_internal(val->children1)) {
			return false;
		}
	}

	return true;
}

// https://dreampuf.github.io/GraphvizOnline/
// http://magjac.com/graphviz-visual-editor/
// https://zhu45.org/posts/2017/May/25/draw-a-neural-network-through-graphviz/
static bool
Value_print_dot(const Value *val, const char *title) {
	assert(Value_invariant(val));
	const char * const preamble = "/* Generated by micrograd.c */\n"
		"digraph {\n"
			"\tnode [shape=Mrecord];\n"
			"\tedge [arrowhead=vee];\n"
			"\tsplines=true;\n"
			"\trankdir=TB;\n"
			"\tlabel=\"%s\";\n"
			"\ttooltip=\"Hover the mouse over a\\nnode to reveal its address.\";\n"
			"\tfontname=Helvetica;\n";

	// TODO: title should be sanitized.
	return printf(preamble, title) >= 0
		&& Value_print_dot_internal(val)
		&& printf("}\n") >= 0;

	return true;
}

#pragma mark Neural Network

// TODO: bias should just be the n+1 weight. This makes iteration easier.
typedef struct Neuron {
	Value **weights;
	unsigned n_weights;
	Value *bias;
} Neuron;

static bool
Neuron_invariant(Neuron *neuron) {
	if (neuron == NULL || neuron->n_weights == 0) {
		return false;
	}

	for (unsigned i = 0; i < neuron->n_weights; i++) {
		if (!Value_invariant(neuron->weights[i])) {
			return false;
		}
	}

	return Value_invariant(neuron->bias);
}

static bool
Neuron_new(Arena alloc[static 1], unsigned n_inputs, Neuron res[static 1]) {
	assert(Arena_invariant(alloc));
	assert(n_inputs > 0);
	assert(res != NULL);

	res->weights = Arena_alloc(alloc, sizeof *res->weights * n_inputs);
	if (res->weights == NULL) {
		return false;
	}
	res->n_weights = n_inputs;
	for (unsigned i = 0; i < n_inputs; i++) {
		res->weights[i] = Value_new(alloc, RNG_uniform_m1p1());
	}
	res->bias = Value_new(alloc, RNG_uniform_m1p1());

	assert(Neuron_invariant(res));
	return true;
}

// NOTE: At every forward step we regenerate the graph. This means that w have
// to store the weights in a different place than the intermediate ones
// calculated by the various forward operations.
// Another option could be to reuse that very handy topological order that we
// already need for the back propagation...

static Value *
Neuron_forward(Arena alloc[static 1], Neuron neuron[static 1], Value *inputs[static neuron->n_weights]) {
	assert(Arena_invariant(alloc));
	assert(Neuron_invariant(neuron));

	Value *res = neuron->bias;

	for (unsigned i = 0; i < neuron->n_weights; i++) {
		res = Value_add(alloc, res, Value_mul(alloc, neuron->weights[i], inputs[i]));
	}

	return Value_tanh(alloc, res);
}

typedef struct Layer {
	Neuron *neurons;
	unsigned n_neurons;
} Layer;

static bool
Layer_invariant(Layer *layer) {
	if (layer == NULL || layer->n_neurons == 0) {
		return false;
	}

	unsigned n_weights = layer->neurons[0].n_weights;
	for (unsigned i = 1; i < layer->n_neurons; i++) {
		if (layer->neurons[i].n_weights != n_weights
			|| !Neuron_invariant(layer->neurons + i)) {
			return false;
		}
	}

	return true;
}

static void
Layer_forward(
		Arena alloc[static 1],
		Layer layer[static 1],
		Value *inputs[static layer->neurons[0].n_weights],
		Value *outputs[static layer->n_neurons]
	) {
	assert(Arena_invariant(alloc));
	assert(Layer_invariant(layer));

	for (unsigned i = 0; i < layer->n_neurons; i++) {
		outputs[i] = Neuron_forward(alloc, layer->neurons + i, inputs);
	}
}

typedef struct MLP {
	Layer *layers;
	unsigned n_layers;
} MLP;

static bool
MLP_invariant(MLP *mlp) {
	if (mlp == NULL || mlp->n_layers == 0) {
		return false;
	}

	for (unsigned i = 0; i < mlp->n_layers; i++) {
		if (!Layer_invariant(mlp->layers + i)) {
			return false;
		}
	}

	return true;
}

static NODISCARD bool
MLP_new(
		Arena alloc[static 1],
		unsigned input_and_layers_size_len,
		unsigned input_and_layers_size[static input_and_layers_size_len],
		MLP res[static 1]
	) {
	assert(Arena_invariant(alloc));
	assert(input_and_layers_size_len >= 2);

	res->layers = Arena_alloc(alloc, sizeof *res->layers * (input_and_layers_size_len - 1));
	res->n_layers = input_and_layers_size_len - 1;
	if (res->layers == NULL) {
		return false;
	}

	for (unsigned layer_index = 1; layer_index < input_and_layers_size_len; layer_index++) {
		res->layers[layer_index - 1].neurons = Arena_alloc(alloc, sizeof (Neuron) * input_and_layers_size[layer_index]);
		res->layers[layer_index - 1].n_neurons = input_and_layers_size[layer_index];
		if (res->layers[layer_index - 1].neurons == NULL) {
			return false;
		}

		for (unsigned i = 0; i < input_and_layers_size[layer_index]; i++) {
			if (!Neuron_new(alloc, input_and_layers_size[layer_index - 1],
				res->layers[layer_index - 1].neurons + i)) {
				return false;
			}
		}
	}

	assert(MLP_invariant(res));
	return true;
}

static NODISCARD bool
MLP_forward(
		Arena alloc[static 1],
		MLP mlp[static 1],
		Value *inputs[static mlp->layers[0].neurons[0].n_weights],
		Value *outputs[static mlp->layers[mlp->n_layers - 1].n_neurons]) {
	assert(Arena_invariant(alloc));
	assert(MLP_invariant(mlp));

	unsigned n_inputs = mlp->layers[0].neurons[0].n_weights;
	unsigned n_outputs = mlp->layers[mlp->n_layers - 1].n_neurons;
	unsigned max = n_inputs;
	for (unsigned i = 1; i < mlp->n_layers; i++) {
		unsigned candidate = mlp->layers[i].neurons[0].n_weights;
		if (candidate > max) {
			max = candidate;
		}
	}

	Value **layer_in = Arena_alloc(alloc, sizeof *layer_in * max),
		**layer_out = Arena_alloc(alloc, sizeof *layer_out * max);
	if ((layer_in == NULL) | (layer_out == NULL)) {
		return false;
	}

	// NOTE: this could be a memcpy.
	for (unsigned i = 0; i < n_inputs; i++) {
		layer_in[i] = inputs[i];
	}

	for (unsigned i = 0; i < mlp->n_layers; i++) {
		Layer_forward(alloc, mlp->layers + i, layer_in, layer_out);
		// NOTE: this could be a swap.
		Value **tmp = layer_in;
		layer_in = layer_out;
		layer_out = tmp;
	}

	// NOTE: this could be a memcpy.
	// Since we always swap layer_in with layer_out the output is in layer_in.
	for (unsigned i = 0; i < n_outputs; i++) {
		outputs[i] = layer_in[i];
	}

	return true;
}

// TODO: add a a way to pretty print the MLP to dot.

#pragma mark Main

int main(void) {

	static unsigned char mem[1 << 16]; // 65 Kibibyte.
	Arena *alloc = &Arena_FROM_ARRAY(mem);

	// printf("sizeof (Value) == %zu\n", sizeof (Value));

	if (false) {
		Value *a = Value_new(alloc, 2.0),
			*b = Value_new(alloc, -3.0),
			*c = Value_new(alloc, 10.0);
		Value *res = Value_add(alloc, a, Value_tanh(alloc, Value_add(alloc, Value_mul(alloc, a, b), c)));
		Value_backprop(alloc, res);
		Value_print_dot(res, "My test");
		if (res == NULL) {
			return 1;
		}
	}

	if (false) {
		Value *x1 = Value_new(alloc, 2.0);
		Value *x2 = Value_new(alloc, 0.0);
		Value *w1 = Value_new(alloc, -3.0);
		Value *w2 = Value_new(alloc, 1.0);
		Value *b  = Value_new(alloc, 6.8813735870195432);
		Value *x1w1 = Value_mul(alloc, x1, w1);
		Value *x2w2 = Value_mul(alloc, x2, w2);
		Value *x1w1x2w2 = Value_add(alloc, x1w1, x2w2);
		Value *n = Value_add(alloc, x1w1x2w2, b);
		Value *o = Value_tanh(alloc, n);
		Value_backprop(alloc, o);
		Value_print_dot(o, "Karpathy test 1");
		if (o == NULL) {
			return 2;
		}
	}

	if (false) {
		Value *a = Value_new(alloc, -2.0);
		Value *b = Value_new(alloc, 3.0);
		Value *d = Value_mul(alloc, a, b);
		Value *e = Value_add(alloc, a, b);
		Value *f = Value_mul(alloc, d, e);
		Value_backprop(alloc, f);
		Value_print_dot(f, "Karpathy test 2");
		if (f == NULL) {
			return 3;
		}
	}

	Arena_reset(alloc);

	if (false) {
		Value *res = Value_div(alloc,
			Value_sub(alloc, Value_new(alloc, 4), Value_new(alloc, 2)),
			Value_new(alloc, 2)
		);
		Value_backprop(alloc, res);
		Value_print_dot(res, "Non-elementary operations");
		if (res == NULL) {
			return 4;
		}
	}

	Arena_reset(alloc);

	if (false) {
		RNG_set_seed(42);
		unsigned layers[] = {3, 4, 4, 1};
		MLP mlp = {};
		if (!MLP_new(alloc, ARRAY_LEN(layers), layers, &mlp)) {
			return 5;
		}

		Value *inputs[] = {Value_new(alloc, 1), Value_new(alloc, 2), Value_new(alloc, 3)};
		Value *outputs[1] = {};

		Arena temp = Arena_from_unused(alloc);
		for (unsigned epoch = 0; epoch < 10; epoch++) {
			MLP_forward(&temp, &mlp, inputs, outputs);
			if (outputs[0] == NULL) {
				return 6;
			}
			if(!Value_backprop(&temp, outputs[0])) {
				return 7;
			}
			Arena_reset(&temp);
		}

		MLP_forward(&temp, &mlp, inputs, outputs);
		if (!Value_backprop(&temp, outputs[0])) {
			return 8;
		}
		Value_print_dot(outputs[0], "Big MLP");
	}

	if (false) {
		RNG_set_seed(42);
		// unsigned layers[] = {3, 1};
		unsigned layers[] = {1, 2, 1};
		MLP mlp = {};
		if (!MLP_new(alloc, ARRAY_LEN(layers), layers, &mlp)) {
			return 5;
		}

		Value *inputs[] = {Value_new(alloc, 1), Value_new(alloc, 2), Value_new(alloc, 3)};
		Value *outputs[1] = {};

		MLP_forward(alloc, &mlp, inputs, outputs);
		if (outputs[0] == NULL) {
			return 9;
		}
		Value_print_dot(outputs[0], "Small MLP"); fflush(stdout);
		if(!Value_backprop(alloc, outputs[0])) {
			return 10;
		}
		Value_print_dot(outputs[0], "Small MLP");
		Arena_reset(alloc);
	}

	Arena_reset(alloc);

	if (true) {
		// Training test.
		RNG_set_seed(42);
		unsigned layers[] = {3, 4, 4, 1};
		MLP mlp = {};
		if (!MLP_new(alloc, ARRAY_LEN(layers), layers, &mlp)) {
			return 5;
		}

		Value *xs[] = {
			Value_new(alloc, 2), Value_new(alloc, 3), Value_new(alloc, -1),
			Value_new(alloc, 3), Value_new(alloc, -1), Value_new(alloc, .5),
			Value_new(alloc, .5), Value_new(alloc, 1), Value_new(alloc, 1),
			Value_new(alloc, 1), Value_new(alloc, 1), Value_new(alloc, -1),
		};
		Value *ys[] = {
			Value_new(alloc, 1), Value_new(alloc, -1), Value_new(alloc, -1), Value_new(alloc, 1),
		};
		Value *ypred[ARRAY_LEN(ys)] = {};

		Arena temp = Arena_from_unused(alloc);
		for (unsigned epoch = 0; epoch < 40; epoch++) {
			// We calculate the forward for the one and only batch.
			for (unsigned i = 0; i < ARRAY_LEN(xs)/3; i++) {
				if (!MLP_forward(&temp, &mlp, xs + (i*3), ypred + i)) {
					return 11;
				}
			}

			// We calculate the loss on the one and olny batch.
			Value *loss =  Value_pow(&temp, Value_sub(&temp, ys[0], ypred[0]), 2);
			for (unsigned i = 1; i < ARRAY_LEN(ys); i++) {
				loss = Value_add(&temp, loss, Value_pow(&temp, Value_sub(&temp, ys[i], ypred[i]), 2));
			}
			if (loss == NULL) {
				return 12;
			}
			printf("epoch %2u loss: %lf\n", epoch, loss->data);

			if(!Value_backprop(&temp, loss)) {
				return 13;
			}

			double learning_rate = 0.05;

			// NOTE: it would be beneficial to allocate the parameters of the
			// neural network in a single array in memory so that iterating over
			// them is easier and faster.
			// NOTE: In the book Deep Learnig by Googfellot et al. they describe
			// (in pseudocode) the backpropagation algorithm that keeps the
			// temporary gradients "off band" from the node graph. This may lead
			// a reduciton in memory usage due to allignment.
			// Updating parameters and zeroing gradients.
			for (unsigned l = 0; l < mlp.n_layers; l++) {
				Layer *layer = mlp.layers + l;
				for (unsigned n = 0; n < layer->n_neurons; n++) {
					Neuron *neuron = layer->neurons + n;
					for (unsigned w = 0; w < neuron->n_weights; w++) {
						if (neuron->weights[w] != NULL) {
							neuron->weights[w]->data -= learning_rate * neuron->weights[w]->grad;
							neuron->weights[w]->grad = 0;
						}
					}
					if (neuron->bias != NULL) {
						neuron->bias->data -= learning_rate * neuron->bias->grad;
						neuron->bias->grad = 0;
					}
				}
			}
			Arena_reset(&temp);
		}
	}

	return 0;
}
