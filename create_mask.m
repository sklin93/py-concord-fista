function M = create_mask(n)
% create edge_adjacency mask

	n_edge = (n*(n-1)/2);
	A = inline_vec2mat(1:n_edge, n);
	M = eye(n_edge, n_edge);
	for i = 1:n,
		edge_idx = A(i,:);
		edge_idx = edge_idx([1:i-1,i+1:n]);
		M(edge_idx, edge_idx) = 1;
	end
    
end

function A = inline_vec2mat(v, n)

	A = zeros(n,n);
	% map to lower triangular matrix
	A(tril(true(size(A)),-1)) = v;
	% add one to diagonal and symmetrize
	A = A + A' + eye(n);

end

