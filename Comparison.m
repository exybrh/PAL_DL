%% Proximal augmented Lagrangian vs. Subgradient Descent
%% Problem settings
n = 30;%50;
m = 3*10*n^2;
%A = eye(n);
M = randn(n,n);
A = orth(M);
norm(A'*A - eye(n))
theta = 0.3;
X = zeros(n,m);
for j = 1:m
    for i = 1:n
        X(i,j) = (rand <= theta) * randn;
    end
end
Y = A*X;
Table = [];
tablerow = [];
%% Algorithms
for r = 1:3
    % generating uniform point on a sphere
    q = randn(1,n);
    s2 = sum(q.^2,2);
    q = q.*repmat(1*(gammainc(s2/2,n/2).^(1/n))./sqrt(s2),1,n);
    q = q/norm(q);
    q = q';
    q_initial = q;
    % Subgradient over manifold
    k = 1;
    epsilon = 1e-3;
    q = q_initial;
    fval = 0;
    lb = inf;
    tic
    while(1)
        subdf = Y*(sign(q'*Y))'/m;
        v = (eye(n) - q*q')*subdf;
        q = (q-v/sqrt(k))/norm(q-v/sqrt(k));
        k = k+1;
        fval = norm(q'*Y,1)/m;
        [r,k]
        if fval <= lb
            lb = fval;
            [maxval,index] = max(abs(q'*A));
            err = min(norm(q - A(:,index)), norm(q + A(:,index)) )
            if err <= epsilon
                break;
            end
        end
        if toc > 30
            break
        end
    end
    tablerow = [index, err, toc, k];
    %% Proximal AL
    epsilon = 1e-3;
    mu = 1e-2;
    iter_M = 0;
    beta = 1;
    rho = 1;
    k = 1;
    lambda = 0;
    q = q_initial;
    tic
    while(1)
        q0 = q;
        L0 = mean( mu*log( ( exp( (Y'*q)/mu) + exp( -(Y'*q)/mu) )/2 )) + lambda*(norm(q) - 1) + rho/2*(norm(q)-1)^2 + beta/2*(q-q0)'*(q-q0);
        % gradient descent
        iter_inner = 0;
        while(1)
            gh = ( 1 - 2./( exp(2*Y'*q/mu) +1) )/m;
            gd = Y*gh + lambda*q/norm(q) + rho*( q - q/norm(q) ) + beta*(q-q0);
            if norm(gd) <= min(epsilon,1/(iter_M+1))
                break;
            end
            % backtracking
            k = 0;
            while(1)
                q_temp = q - gd/2^k;
                L1 = mean( mu*log( ( exp( (Y'*q_temp)/mu ) + exp( -(Y'*q_temp)/mu) )/2 )) + lambda*(norm(q_temp) - 1) + rho/2*(norm(q_temp)-1)^2 + beta/2*(q_temp-q0)'*(q_temp-q0);
                if L1 - L0 <= - (1/2)*(1/2^k)*(gd'*gd)
                    L0 = L1;
                    q = q_temp;
                    break;
                end
                k = k+1;
            end
            iter_inner = iter_inner + 1;
        end
        lambda = lambda + rho*(norm(q)-1);
        iter_M = iter_M + 1;
        if (beta*norm(q-q0) <= epsilon && abs(norm(q) - 1) <= epsilon) || toc > 300
            break
        end
    end
    toc
    [maxval1,index1] = max(abs(q'*A));
    err1 = min(norm(q - A(:,index)), norm(q + A(:,index)) );
    tablerow = [tablerow, index1, err1, toc, iter_M];
    %% Proximal AL with small beta and large rho
    epsilon = 1e-3;
    mu = 1e-2;
    iter_M = 0;
    beta = .1;
    rho = 10;
    k = 1;
    lambda = 0;
    q = q_initial;
    tic
    while(1)
        q0 = q;
        L0 = mean( mu*log( ( exp( (Y'*q)/mu) + exp( -(Y'*q)/mu) )/2 )) + lambda*(norm(q) - 1) + rho/2*(norm(q)-1)^2 + beta/2*(q-q0)'*(q-q0);
        % gradient descent
        iter_inner = 0;
        while(1)
            gh = ( 1 - 2./( exp(2*Y'*q/mu) +1) )/m;
            gd = Y*gh + lambda*q/norm(q) + rho*( q - q/norm(q) ) + beta*(q-q0);
            if norm(gd) <= min(epsilon,1/(iter_M+1))
                break;
            end
            % backtracking
            k = 0;
            while(1)
                q_temp = q - gd/2^k;
                L1 = mean( mu*log( ( exp( (Y'*q_temp)/mu ) + exp( -(Y'*q_temp)/mu) )/2 )) + lambda*(norm(q_temp) - 1) + rho/2*(norm(q_temp)-1)^2 + beta/2*(q_temp-q0)'*(q_temp-q0);
                if L1 - L0 <= - (1/2)*(1/2^k)*(gd'*gd)
                    L0 = L1;
                    q = q_temp;
                    break;
                end
                k = k+1;
            end
            iter_inner = iter_inner + 1;
        end
        lambda = lambda + rho*(norm(q)-1);
        iter_M = iter_M + 1;
        if (beta*norm(q-q0) <= epsilon && abs(norm(q) - 1) <= epsilon) || toc > 300
            break
        end
    end
    toc
    [maxval2,index2] = max(abs(q'*A));
    err2 = min(norm(q - A(:,index)), norm(q + A(:,index)) );
    tablerow = [tablerow, index2, err2, toc, iter_M];
    %% Proximal AL with smallest beta and largest rho
    epsilon = 1e-3;
    mu = 1e-2;
    iter_M = 0;
    beta = .01;
    rho = 100;
    k = 1;
    lambda = 0;
    q = q_initial;
    tic
    while(1)
        q0 = q;
        L0 = mean( mu*log( ( exp( (Y'*q)/mu) + exp( -(Y'*q)/mu) )/2 )) + lambda*(norm(q) - 1) + rho/2*(norm(q)-1)^2 + beta/2*(q-q0)'*(q-q0);
        % gradient descent
        iter_inner = 0;
        while(1)
            gh = ( 1 - 2./( exp(2*Y'*q/mu) +1) )/m;
            gd = Y*gh + lambda*q/norm(q) + rho*( q - q/norm(q) ) + beta*(q-q0);
            if norm(gd) <= min(epsilon,1/(iter_M+1))
                break;
            end
            % backtracking
            k = 0;
            while(1)
                q_temp = q - gd/2^k;
                L1 = mean( mu*log( ( exp( (Y'*q_temp)/mu ) + exp( -(Y'*q_temp)/mu) )/2 )) + lambda*(norm(q_temp) - 1) + rho/2*(norm(q_temp)-1)^2 + beta/2*(q_temp-q0)'*(q_temp-q0);
                if L1 - L0 <= - (1/2)*(1/2^k)*(gd'*gd)
                    L0 = L1;
                    q = q_temp;
                    break;
                end
                k = k+1;
            end
            iter_inner = iter_inner + 1;
        end
        lambda = lambda + rho*(norm(q)-1);
        iter_M = iter_M + 1;
        if (beta*norm(q-q0) <= epsilon && abs(norm(q) - 1) <= epsilon) || toc > 300
            break
        end
    end
    toc
    [maxval3,index3] = max(abs(q'*A));
    err3 = min(norm(q - A(:,index)), norm(q + A(:,index)) );
    tablerow = [tablerow, index3, err3, toc, iter_M];
    Table = [Table;tablerow];
end