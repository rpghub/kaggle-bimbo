###   lagged xgboost model

install.packages('data.table')
install.packages('xgboost')
install.packages('Matrix')
install.packages('stringr')

library(data.table)
library(xgboost)
library(Matrix)
library(stringr)

# parameters
data_loc <- '/home/ubuntu/data'
saveloc <- '/home/ubuntu/data'
smpl_size <- 18E4
max_week <- 9

# eval metric
f_error <- function(y, p) {
  sqrt(mean((log(p+1) - log(y + 1))^2))
}

# unzip and read data
setwd(data_loc)
unzip('cliente_tabla.csv.zip')
unzip('producto_tabla.csv.zip')
unzip('test.csv.zip')
unzip('town_state.csv.zip')
unzip('train.csv.zip')

# process product and town data -----------------------------------------------
dat <- fread('train.csv')
test <- fread('test.csv')
prod <- fread('producto_tabla.csv')
town <- fread('town_state.csv')
client <- fread('cliente_tabla.csv')

# rename and combine main data
setnames(dat, c('week', 'sales_id', 'channel_id', 'route_id', 'client_id'
                , 'product_id', 'sale_qt', 'sale', 'return_qt', 'return'
                , 'demand'))

setnames(test, c('id', 'week', 'sales_id', 'channel_id', 'route_id', 'client_id'
                 , 'product_id'))

dat[, id := seq_len(.N)]
dat <- rbind(dat, test, fill = TRUE)
rm(test)

# add extra groupings based on tables -----------------------------------------
# add fields to product table
prod[, weight_g := as.integer(str_extract(NombreProducto, '(([0-9])+(?=[gG]))'))]
prod[, weight_kg := as.integer(str_extract(NombreProducto, '(([0-9])+(?=[Kk]g))'))]
prod[, weight_ml := as.integer(str_extract(NombreProducto, '([0-9])+(?=( ?ml))'))]
prod[, pieces := as.integer(str_extract(NombreProducto, '([0-9])+(?=[pP]q? )'))]
prod[, brand := str_extract(NombreProducto, '\\w+(?=(\\W)\\w+$)')]

# get base name
prod[!is.na(pieces), end_base := regexpr('([0-9])+([Pp]q?)', NombreProducto)]
prod[(!is.na(weight_g) | !is.na(weight_kg) | !is.na(weight_ml)) & is.na(pieces)
     , end_base := regexpr('([0-9])+( ?[kKgmlML])', NombreProducto)]
prod[, base_name := str_sub(NombreProducto, 1, end_base - 2)]
prod[, end_base := NULL]

prod[is.na(weight_g), weight_g := as.integer(weight_kg) * 1000L]
prod[is.na(weight_g), weight_g := 0]
prod[is.na(weight_ml), weight_ml := 0]
prod[, weight_kg := NULL]
prod[is.na(pieces), pieces := 1]
setnames(prod, 'Producto_ID', 'product_id')

# brands with few obs into all other
prod[, brand_cnt := .N, brand]
prod[brand_cnt < 5, brand := 'XXX']
prod[, brand_cnt := NULL]
prod[, prod_grp_id := .GRP, base_name]
prod[, brand_id := .GRP, brand]
prod[is.na(base_name), base_name := 'OTHER']

# add towns
setnames(town, 'Agencia_ID', 'sales_id')
town[, state_id := .GRP, State]
town[, town_id := .GRP, Town]

# client ids
client[, seq := seq_len(.N), Cliente_ID]
client <- client[seq == 1]  # randomly choose one name
client[, seq := NULL]
client[, name_cnt := .N, NombreCliente]
client[name_cnt < 2, grp_name := 'All Other']
client[name_cnt >= 2, grp_name := NombreCliente]
client[, client_grp_id := .GRP, grp_name]
setnames(client, 'Cliente_ID', 'client_id')

# merge in prod
setkey(dat, product_id)
setkey(prod, product_id)
dat <- prod[dat]

# merge in town
setkey(dat, sales_id)
setkey(town, sales_id)
dat <- town[dat]

# merge in client
setkey(dat, client_id)
setkey(client, client_id)
dat <- client[, .(client_id, client_grp_id)][dat]
dat[is.na(client_grp_id), client_grp_id := 0]
rm(client, prod, town)

# functions for fitting -------------------------------------------------------

# function to add lagged features
f_lag <- function(id, week, val, name = '', min_lag, max_lag, fill = NA) {
  min_week <- min(week)
  max_week <- max(week)
  tmp <- data.table(id = id, week = week, val = val)
  tmp <- tmp[, .(val = mean(val), cnt = .N), .(week, id)]
  tmp_full <- tmp[, .N, id][, .(week = seq(min_week, max_week)), id]
  setkey(tmp_full, id, week)
  setkey(tmp, id, week)
  tmp_full <- tmp_full[!tmp]
  tmp_full[, cnt := 0]
  tmp <- rbind(tmp_full, tmp, fill = TRUE)
  rm(tmp_full)
  tmp[, match_week := week]
  for (i in seq(min_lag, max_lag)) {
    tmp_join <- tmp[, .(id, match_week = match_week + i, val, cnt
                        , dummy = 1 * (cnt > 0))]
    setkey(tmp_join, id, match_week)
    tmp_join[, valcum := cumsum(ifelse(is.na(val), 0, val)), id]
    tmp_join[, valcumcnt := cumsum(dummy), id]
    tmp_join[, dummy := NULL]
    tmp_join[, cnt := NULL]
    tmp_join[, val := NULL]
    
    tmp_join_lag <- 
      tmp_join[, .(id, match_week = match_week + 2, valcumrng = valcum
                   , valcumrngcnt = valcumcnt)]
    setkey(tmp_join_lag, id, match_week)
    tmp_join <- tmp_join_lag[tmp_join]
    tmp_join[is.na(valcumrng), valcumrng := 0]
    tmp_join[is.na(valcumrngcnt), valcumrngcnt := 0]
    tmp_join[, valcumrng := (valcum - valcumrng) / (valcumcnt - valcumrngcnt)]
    tmp_join[, valcumrngcnt := valcumcnt - valcumrngcnt]
    tmp_join[, valcum := valcum / valcumcnt]
    
    tmp_join[, valcumrngcnt := NULL]
    setnames(tmp_join, c('valcum', 'valcumcnt', 'valcumrng')
             , c( paste('valcum', name, i, sep = '')
                  , paste('valcumcnt', name, i, sep = '')
                  , paste('valcumrng', name, i, sep = '')))
    setkey(tmp_join, id, match_week)
    setkey(tmp, id, match_week)
    tmp <- tmp_join[tmp]
  }
  tmp[, match_week := NULL]
  tmp <- tmp[cnt > 0]
  tmp[, cnt := NULL]
  tmp[, val := NULL]
  if (!is.na(fill)) {
    cols <- colnames(tmp)
    tmp[, (cols) := lapply(.SD, function(x) ifelse(is.na(x), fill, x))
        , .SDcols = cols]
  }
  tmp
}

f_add_lag <- function(tbl, val_name, id_name, lag_name = '', min_lag, max_lag, fill = NA) {
  tbl <- copy(tbl)
  if (length(val_name) != length(id_name) | length(val_name) != length(lag_name)) {
    stop('lengths of val_name and id_name do not match')
  }
  for (i in 1:length(val_name)) {
    print(i)
    setnames(tbl, c(val_name[i], id_name[i]), c('tmp_val', 'tmp_id'))
    tmp <- f_lag(tbl[, tmp_id], tbl[, week], tbl[, tmp_val], lag_name[i]
                 , min_lag, max_lag, fill)
    setkey(tmp, id, week)
    setkey(tbl, tmp_id, week)
    chk <- nrow(tbl)
    tmp <- tbl[, .(tmp_id, week)][tmp]
    if (chk != nrow(tmp)) {
      stop('rows not the same')
    }
    cols <- setdiff(colnames(tmp), c('tmp_id', 'week'))
    lapply(cols, function(x) tbl[, (x) := tmp[[x]]])
    setnames(tbl, c('tmp_val', 'tmp_id'), c(val_name[i], id_name[i]))
  }
  tbl
} 

f_xgb_fit <- function(f_grp) {
  print(f_grp)
  dat_smpl <- dat[fit_grp == f_grp]
  # create ids
  dat_smpl[, base_id := .GRP, base_name]
  dat_smpl[, sales_prod := .GRP, .(sales_id, product_id)]
  dat_smpl[, sales_prod_client := .GRP, .(sales_id, product_id, client_id)]
  dat_smpl[, sales_client := .GRP, .(sales_id, client_id)]
  dat_smpl[, sales_rte := .GRP, .(sales_id, route_id)]
  dat_smpl[, prod_client := .GRP, .(client_id, product_id)]
  dat_smpl[, prod_rte := .GRP, .(product_id, route_id)]
  dat_smpl[, prod_town := .GRP, .(product_id, town_id)]
  dat_smpl[, client_town := .GRP, .(client_id, town_id)]
  dat_smpl[, prod_state := .GRP, .(product_id, state_id)]
  dat_smpl[, client_state := .GRP, .(client_id, state_id)]
  dat_smpl[, base_client := .GRP, .(client_id, base_id)]
  dat_smpl[, base_rte := .GRP, .(base_id, route_id)]
  dat_smpl[, base_town := .GRP, .(base_id, town_id)]
  dat_smpl[, base_state := .GRP, .(base_id, state_id)]
  dat_smpl[, sales_base := .GRP, .(sales_id, base_id)]
  dat_smpl[, client_rte := .GRP, .(client_id, route_id)]
  
  dat_smpl[, client_gpr_state := .GRP, .(client_grp_id, state_id)]
  dat_smpl[, base_client_grp := .GRP, .(client_grp_id, base_id)]
  dat_smpl[, client_grp_brand := .GRP, .(client_grp_id, brand_id)]
  dat_smpl[, prod_client_grp := .GRP, .(client_grp_id, product_id)]
  dat_smpl[, sales_prod_client_grp := .GRP, .(sales_id, product_id, client_grp_id)]
  dat_smpl[, sales_base_client_grp := .GRP, .(sales_id, base_id, client_grp_id)]
  dat_smpl[, client_grp_brand := .GRP, .(client_grp_id, brand_id)]
  dat_smpl[, client_channel := .GRP, .(client_id, channel_id)]
  dat_smpl[, prod_channel := .GRP, .(product_id, channel_id)]
  
  dat_smpl[, log_demand := log(demand + 1)]
  dat_smpl[, log_return := log(return_qt + 1)]
  dat_smpl[, log_sales := log(sale_qt + 1)]
  
  
  dat_smpl[, client_itemcnt := .N, .(client_id, week)]
  dat_smpl[, client_sales_itemcnt := .N, .(client_id, sales_id, week)]
  dat_smpl[, sales_itemcnt := .N, .(sales_id, week)]
  dat_smpl[, prod_sales_itemcnt := .N, .(product_id, sales_id, week)]
  dat_smpl[, state_prod_itemcnt := .N, .(state_id, product_id, week)]
  dat_smpl[, town_prod_itemcnt := .N, .(town_id, product_id, week)]
  dat_smpl[, prod_rte_itemcnt := .N, .(product_id, route_id, week)]
  dat_smpl[, sales_rte_itemcnt := .N, .(sales_id, route_id, week)]
  
  
  # create table of lag features to create
  lag_feature <-
    data.table(id = setdiff(colnames(dat_smpl)
                            , c('Town', 'State', 'NombreProducto', 'weight_g'
                                , 'weight_ml', 'pieces', 'brand', 'base_name'
                                , 'week', 'sale_qt', 'sale', 'return_qt'
                                , 'return', 'demand', 'id', 'log_demand', 'log_return'
                                , 'log_sales', 'client_itemcnt', 'client_sales_itemcnt'
                                , 'sales_itemcnt', 'prod_sales_itemcnt'
                                , 'state_prod_itemcnt', 'town_prod_itemcnt'
                                , 'prod_rte_itemcnt', 'sales_rte_itemcnt')))
  lag_feature <- lag_feature[, .(val = c('log_demand', 'log_return')), id]
  
  lag_feature <- 
    rbind(lag_feature
          , data.table(id = c('client_id', 'sales_client', 'sales_id'
                              , 'sales_prod', 'prod_state', 'prod_town'
                              , 'prod_rte', 'sales_rte')
                       , val = c('client_itemcnt', 'client_sales_itemcnt'
                                 , 'sales_itemcnt', 'prod_sales_itemcnt'
                                 , 'state_prod_itemcnt', 'town_prod_itemcnt'
                                 , 'prod_rte_itemcnt', 'sales_rte_itemcnt')))
  
  lag_feature[val == 'log_demand', lag_name := paste(id, '_d', sep = '')]
  lag_feature[val == 'log_return', lag_name := paste(id, '_r', sep = '')]
  lag_feature[val == 'log_sales', lag_name := paste(id, '_s', sep = '')]
  lag_feature[is.na(lag_name), lag_name := val]
  
  dat_smpl <- f_add_lag(dat_smpl, lag_feature[, val], lag_feature[, id]
                        , lag_feature[, lag_name], 2, 2, -10)
  
  # fit on data that has 1 or more lags
  dat_smpl <- dat_smpl[week >= 7]
  
  # set up data for xgb
  cols <- setdiff(colnames(dat_smpl), c('Town', 'State', 'NombreProducto', 'brand'
                                        , 'base_name', 'week', 'sale_qt', 'sale'
                                        , 'return_qt', 'return', 'demand', 'id'
                                        , 'seq_id', 'log_demand', 'log_return'
                                        , 'log_sales'
                                        , 'client_itemcnt', 'client_sales_itemcnt'
                                        , 'sales_itemcnt', 'prod_sales_itemcnt'
                                        , 'state_prod_itemcnt', 'town_prod_itemcnt'
                                        , 'prod_rte_itemcnt', 'sales_rte_itemcnt'
                                        , 'client_id', 'client_grp_id', 'sales_id', 'state_id'
                                        , 'town_id', 'product_id', 'prod_grp_id', 'brand_id'
                                        , 'channel_id', 'route_id', 'fit_grp', 'base_id'
                                        , 'sales_prod', 'sales_prod_client', 'sales_client'
                                        , 'sales_rte', 'prod_client', 'prod_rte', 'prod_town'
                                        , 'client_town', 'prod_state', 'client_state'
                                        , 'client_rte', 'client_grp_state', 'base_client_grp'
                                        , 'client_grp_brand', 'prod_client_grp', 'sales_prod_client_grp'
                                        , 'sales_base_client_grp', 'client_channel', 'prod_channel'
                                        , 'base_client', 'base_rte', 'base_town', 'base_state'
                                        , 'sales_base', 'client_grp_state'))
  
  
  dat_smpl[, smpl_id := seq_len(.N)]
  mat <- Matrix(as.matrix(dat_smpl[, cols, with = FALSE]), sparse = TRUE)
  
  dtrain <- xgb.DMatrix(data = mat[dat_smpl[week <= max_week, smpl_id], ]
                        , label = dat_smpl[week <= max_week, log(demand + 1)])
  
  watchlist <- list(train = dtrain)
  
  
  fit <- xgb.train(data = dtrain, watchlist = watchlist
                   , nrounds = 400
                   , print.every.n = 10
                   , eval_metric = 'rmse'
                   , objective = 'reg:linear'
                   , eta = .075
                   , subsample = .5
                   , max.depth = 7
  )
  
  pred <- predict(fit, mat[dat_smpl[week > max_week, smpl_id], ])
  write.csv(dat_smpl[week > max_week, .(id, pred = exp(pred) - 1)]
            , paste('xgb_lag_', f_grp, '.csv', sep = '')
            , row.names = FALSE)
}

setwd(saveloc)
clients <- unique(dat[, .(client_id)])
n_grp <- ceiling(nrow(clients) / smpl_size)
clients[, fit_grp := sample.int(n_grp, .N, replace = TRUE)]
clients[, fit_grp := .GRP, fit_grp]
suppressWarnings(dat[, fit_grp := NULL])
setkey(clients, client_id)
setkey(dat, client_id)
dat <- dat[clients]
rm(clients)
lapply(seq_len(n_grp), f_xgb_fit)

# load saved fit groups
# setwd('C:/DATA/Kaggle/Bimbo/Source Data/submissions/post comp')
# grps <- fread('dat grps.csv')
# setkey(grps, id)
# setkey(dat, id)
# dat <- dat[grps]
# rm(grps)
# n_grp <- dat[, uniqueN(fit_grp)]
# lapply(seq(14, n_grp), f_xgb_fit)

f <- paste('xgb_lag_', seq_len(n_grp), '.csv', sep = '')
preds <- do.call('rbind', lapply(f, fread)) 
preds[pred < 0, pred := 0]
rm(f)

setnames(preds, c('id', 'Demanda_uni_equil'))
write.csv(preds, 'xgb post comp lag model.csv', row.names = FALSE)





