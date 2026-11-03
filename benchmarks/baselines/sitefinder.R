#!/usr/bin/env Rscript
# Full-forest formula from the authors' app/app.R; batch I/O added here.
# Source: https://github.com/lynsiewarr/14-3-3-Site-Finder
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 3) stop('Usage: sitefinder.R train TRAIN_CSV MODEL_RDS | score MODEL_RDS FEATURES_CSV OUTPUT_CSV')
suppressPackageStartupMessages(library(randomForest))
seed <- 20261025L
if (args[1] == 'train') {
  dat <- read.csv(args[2], header=TRUE)
  dat <- dat[dat$Amino.Acid %in% c('S','T'), ]
  dat$Response <- as.factor(dat$Response)
  dat$Amino.Acid <- as.factor(dat$Amino.Acid)
  dat$normalization <- as.numeric(dat$normalization)
  dat$normalization[is.na(dat$normalization)] <- 0
  stopifnot(identical(levels(dat$Response), c('0','1')))
  set.seed(seed)
  full <- randomForest(Response~Amino.Acid+Iupred.score+ANN+PSSM+SVM+normalization,
                       data=dat,mtry=2,ntree=500,importance=TRUE,na.action=na.roughfix)
  dir.create(dirname(args[3]), recursive=TRUE, showWarnings=FALSE)
  saveRDS(list(full=full,seed=seed,
               amino_levels=levels(dat$Amino.Acid)),args[3])
  capture.output(sessionInfo(),file=paste0(args[3],'.session.txt'))
  cat('Training rows:',nrow(dat),'positive:',sum(dat$Response=='1'),'negative:',sum(dat$Response=='0'),'\n')
} else if (args[1] == 'score') {
  if (length(args) != 4) stop('Specify MODEL_RDS FEATURES_CSV OUTPUT_CSV')
  models <- readRDS(args[2])
  dat <- read.csv(args[3],header=TRUE,check.names=FALSE,stringsAsFactors=FALSE)
  fields <- c('Amino.Acid','Iupred.score','ANN','PSSM','SVM','normalization')
  stopifnot(all(fields %in% names(dat)),!anyDuplicated(dat$sample_id),all(dat$status=='ready'))
  dat$Amino.Acid <- factor(dat$Amino.Acid,levels=models$amino_levels)
  stopifnot(all(complete.cases(dat[,fields])))
  result <- dat[,c('sample_id','uniprot_id','site_1based','label')]
  result$score_sitefinder <- predict(models$full,dat,type='prob')[,'1']
  dir.create(dirname(args[4]),recursive=TRUE,showWarnings=FALSE)
  write.csv(result,args[4],row.names=FALSE)
  cat('Scored',nrow(result),'sites.\n')
} else stop('Unknown mode')
