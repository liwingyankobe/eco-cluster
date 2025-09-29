# Lotka-Volterra interaction matrix generation for clusters,
# complete solutions at finite size s, non-critical alpha

# Code complementing paper "A minimal model of self-organized clusters
# with phase transitions in ecological communities" by Li, Kardar, Feng,Taylor

# Translated and expanded slightly from original WT Mathematica code (with some help from Claude)

# Version 9/29/25, including documentation

# Usage:

# julia complete.jl size alpha k output-file (alpha-increment iterations)
# (optional arguments in parentheses)

# Size: integer parameter S
# alpha: float parameter \alpha
# k: integer parameter K
# output-file: string (filename)

# alpha-increment, increment: optional float, integer parameters;
# loops through increment times, increasing alpha by alpha-increment each time.

# Example:

# julia complete.jl 12 1.1 2 test-run.txt

# produces file test-run-12-1.1-2.exe:

# (12, 1.1, 2, 31)
# lists parameters, followed by number of distinct stable uninvadable
# equilibria for cluster model with those parameters.

# when run with optional arguments produces one file with multiple
#  lines of the above form

# **Note: does not compute accurately at critical values:
# alpha = 1.0, 0.5,...  since matrix is not invertible at those values**

# note(not implemented): code could be made slightly faster by passing
#  ss to invasion  check rather than re-computing

using LinearAlgebra

# Step function interaction (type 4)
# gives desired interaction between species
function f4(x, aa=1.0, k=2)
    if x == 0
        return 1.0
    elseif abs(x) < k + 1
        return aa
    else
        return 0.0
    end
end

# Sum over periodic images
function total1(x, i, a, n, aa, k)
    result = 0.0
    for j in -n:n
        result += f4(x - j * a, aa, k)
    end
    return result
end

# Potential function wrapper
function potential1(d, size, n=10, i=4, aa=1.0, k=2)
    return total1(d, i, size, n, aa, k)
end

# Generate interaction matrix
function interaction_matrix(n, aa, k)
    matrix = zeros(Float64, n, n)
    
    for i in 0:(n-1)
        for j in 0:(n-1)
            # Calculate distance on circle (minimum of forward and backward distance)
            dist = min(abs(i - j), n - abs(i - j))
            matrix[i+1, j+1] = potential1(dist, n, 10, 4, aa, k)
        end
    end
    
    return matrix
end


# Helper function: extract submatrix for given subset
function subset_matrix(m, set)
    n = length(set)
    submat = zeros(Float64, n, n)
    for i in 1:n
        for j in 1:n
            submat[i, j] = m[set[i], set[j]]
        end
    end
    return submat
end

# Check if all elements are positive
function is_positive(vec)
    return all(x -> x > 0, vec)
end

# Replace function: creates full-size vector from subset values
function replace_vector(size, set, numbers)
    result = zeros(Float64, size)
    for (i, pos) in enumerate(set)
        result[pos] = numbers[i]
    end
    return result
end

# Check invasion resistance
function check_invasion(m, set)
    # Get subset matrix and solve for equilibrium
    ss = subset_matrix(m, set)
    l = length(set)
    lm = size(m, 1)
    
    # Solve for equilibrium densities: ss * pp = 1
    try
        pp = ss \ ones(Float64, l)
        
        # Create full population vector
        pa = replace_vector(lm, set, pp)
        
        # Calculate invasion fitness: r0 = 1 - m * pa
        r0 = ones(Float64, lm) - m * pa
        
        # Check invasion resistance: all species either present or have negative growth
        invasion_resistant = true
        invasion_info = Bool[]
        
        for i in 1:lm
            if i in set
                # Species is present - automatically satisfied
                push!(invasion_info, true)
            else
                # Species is absent - must have negative growth rate
                resistant = r0[i] < 0
                push!(invasion_info, resistant)
                if !resistant
                    invasion_resistant = false
                end
            end
        end
        
        return invasion_resistant, invasion_info
        
    catch e
        # Matrix inversion failed
        return false, Bool[]
    end
end

# Main stability check for a given subset
function is_stable(m, set)
    if isempty(set)
        return false
    end
    
    # Get subset matrix
    ss = subset_matrix(m, set)
    l = length(set)
    
    try
        # Check 1: Positive equilibrium densities
        pp = ss \ ones(Float64, l)
        if !is_positive(pp)
            return false
        end
        
        # Check 2: Local stability (all eigenvalues positive)
        eigenvals = real.(eigvals(ss))
        if !is_positive(eigenvals)
            return false
        end
        
        # Check 3: Invasion resistance
        invasion_stable, _ = check_invasion(m, set)
        if !invasion_stable
            return false
        end
        
        return true
        
    catch e
        # Any numerical error means unstable
        return false
    end
end


# Find all stable subsets - equivalent to stable[m_] in Mathematica
function find_all_stable(m)
    n = size(m, 1)
    stable_sets = Vector{Int}[]
    
    # Generate all non-empty subsets
    for i in 1:(2^n - 1)
        # Convert integer to subset using bit representation
        subset = Int[]
        for j in 1:n
            if (i >> (j-1)) & 1 == 1
                push!(subset, j)
            end
        end
        
        # Check if this subset is stable
        if is_stable(m, subset)
            push!(stable_sets, subset)
        end
    end
    
    return stable_sets
end


# Command line interface
function main()
    if length(ARGS) < 3
        println("Usage: julia script.jl size aa k [output-file-name] [aainc] [aanum]")
        println("  size, aa, k: required parameters")
        println("  output-file-name: optional, defaults to 'complete-output'")
        println("  aainc: optional aa increment")
        println("  aanum: optional number of aa values")
        return
    end
    
    # Parse required arguments
    size = parse(Int, ARGS[1])
    aa = parse(Float64, ARGS[2])
    k = parse(Int, ARGS[3])
    
    # Parse optional arguments
    output_base = length(ARGS) >= 4 ? ARGS[4] : "complete-output"
    aainc = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : nothing
    aanum = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : nothing
    
    # Determine if we're doing a loop or single run
    if aainc !== nothing && aanum !== nothing
        # Loop case
        filename = "$(output_base)-$(size)-$(k).txt"
        println("Running aa loop: start=$aa, increment=$aainc, count=$aanum")
        println("Output file: $filename")
        
        open(filename, "w") do file
            for i in 0:(aanum-1)
                current_aa = aa + i * aainc
                println("Computing for aa = $current_aa...")
                
                M = interaction_matrix(size, current_aa, k)
                stable_sets = find_all_stable(M)
                
                println(file, "($size, $current_aa, $k, $(length(stable_sets)))")
                flush(file)  # Ensure output is written immediately
            end
        end
    else
        # Single run case
        filename = "$(output_base)-$(size)-$(aa)-$(k).txt"
        println("Single run: size=$size, aa=$aa, k=$k")
        println("Output file: $filename")
        
        M = interaction_matrix(size, aa, k)
        stable_sets = find_all_stable(M)
        
        open(filename, "w") do file
            println(file, "($size, $aa, $k, $(length(stable_sets)))")
        end
    end
    
    println("Complete!")
end

# Run main if called from command line
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) > 0
        main()
    else
        # Run tests if no arguments provided
	# tests removed from this version
    end
end