classdef DataLoader1DNested
  %----------------------------------------------------------------------
  properties
    % j is an UNDIRECTED potential neighbor of i if i is a potential neighbor
    % of j or j is a potential neighbor of i.

    % (n x 1) cell where n is the number of images
    % cell{i} contains a concatenation of score matrices S_{ij}
    % where i < j and j is an UNDIRECTED potential neighbor of i.
    S
    % (n x 1) cell containing the set of UNDIRECTED potential neighbors of 
    % each image.
    neighbors
    % used in case only a subset of proposals is considered in each image 
    proposal_indices
    % adjacent matrix built based on neighbors
    adj 
    % (n x n) sparse matrix, S_{ij} = S{i}{neighbor_positions(i,j)} if
    % i < j and j is an UNDIRECTED potential neighbor of i.
    neighbor_positions
  end

  %----------------------------------------------------------------------
  methods
    %------------------------
    % constructor
    function self = DataLoader1DNested(scores)
      self.S = scores.S;
      self.neighbors = scores.e;
      if ~isfield(scores, 'proposal_indices')
        self.proposal_indices = [];
      else
        self.proposal_indices = scores.proposal_indices;
      end

      n = size(self.S, 1);
      % sparse adjacent matrix
      self.adj = sparse(n,n);
      for i = 1:n 
        self.adj(i, self.neighbors{i}) = 1;
      end

      self.neighbor_positions = sparse(n,n);
      for i = 1:n
        % get neighbors of i that have score matrices stored in S{i}
        % make sure that indices of these neighbors are in increasing order
        valid_neighbors = self.neighbors{i}(self.neighbors{i} > i);
        if numel(valid_neighbors) > 1
          assert(all(valid_neighbors(2:end) > valid_neighbors(1:end-1)));
        end
        self.neighbor_positions(i, valid_neighbors) = 1:numel(valid_neighbors);
      end
    end % function

    %------------------------
    % method to get scores
    function res = get_S(self, i, j)
      if i == j
        error('i and j must be different...');
      end

      if self.adj(i,j) == 1 | self.adj(j,i) == 1
        if i < j
          res = self.S{i}{self.neighbor_positions(i,j)};
        elseif i > j
          res = transpose(self.S{j}{self.neighbor_positions(j,i)});
        end
      else 
        res = [];
      end

      if ~isempty(self.proposal_indices) & ~isempty(res)
        res = res(self.proposal_indices{i}, self.proposal_indices{j});
      end

    end % function

  end % method

end % class